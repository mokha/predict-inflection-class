import pickle
import torch
import torch.nn as nn
import pytorch_lightning as L
from sklearn.preprocessing import LabelEncoder
from torchmetrics.classification import MulticlassF1Score
import sentencepiece as spm


class LabelEncoderManager:
    """
    Manages label encoders for POS and Contlex labels, enabling fitting, transforming,
    and inverse transforming of labels.
    """

    def __init__(self):
        self.pos_encoder = LabelEncoder()
        self.contlex_encoders = {}

    def fit(self, pos_labels, contlex_labels):
        """
        Fits label encoders for POS labels and corresponding Contlex labels.

        Args:
            pos_labels: List of POS labels.
            contlex_labels: List of Contlex labels associated with POS labels.
        """
        self.pos_encoder.fit(pos_labels)
        unique_pos = set(pos_labels)
        for pos_class in unique_pos:
            contlex_for_pos = [
                contlex_labels[i]
                for i in range(len(pos_labels))
                if pos_labels[i] == pos_class
            ]
            encoder = LabelEncoder()
            encoder.fit(contlex_for_pos)
            self.contlex_encoders[pos_class] = encoder

    def transform_pos(self, pos_labels):
        """Transforms POS labels using the fitted POS encoder."""
        return self.pos_encoder.transform(pos_labels)

    def inverse_transform_pos(self, encoded_pos_labels):
        """Inverse transforms encoded POS labels."""
        return self.pos_encoder.inverse_transform(encoded_pos_labels)

    def transform_contlex(self, pos_labels, contlex_labels):
        """Transforms Contlex labels using corresponding encoders based on POS."""
        encoded_contlex_labels = []
        for pos, contlex in zip(pos_labels, contlex_labels):
            encoder = self.contlex_encoders.get(pos)
            if encoder is not None:
                encoded_contlex_labels.append(encoder.transform([contlex])[0])
            else:
                raise ValueError(f"No contlex encoder found for POS class: {pos}")
        return encoded_contlex_labels

    def inverse_transform_contlex(self, pos_labels, encoded_contlex_labels):
        """Inverse transforms encoded Contlex labels using corresponding encoders."""
        contlex_labels = []
        for pos, enc_contlex in zip(pos_labels, encoded_contlex_labels):
            encoder = self.contlex_encoders.get(pos)
            if encoder is not None:
                contlex_labels.append(encoder.inverse_transform([enc_contlex])[0])
            else:
                raise ValueError(f"No contlex encoder found for POS class: {pos}")
        return contlex_labels

    def save_encoders(self, path):
        """Saves the encoders to a file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pos_encoder": self.pos_encoder,
                    "contlex_encoders": self.contlex_encoders,
                },
                f,
            )

    def load_encoders(self, path):
        """Loads the encoders from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.pos_encoder = data["pos_encoder"]
            self.contlex_encoders = data["contlex_encoders"]


class SharedEmbeddingTransformer(L.LightningModule):
    """
    A shared embedding transformer model for POS and Contlex prediction.
    """

    def __init__(
        self,
        pos_num_classes,
        contlex_output_map,
        vocab_size,
        embed_size=96,
        hidden_size=128,
        num_layers=2,
        nhead=4,
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=32,
    ):
        super(SharedEmbeddingTransformer, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out_pos = nn.Linear(embed_size, pos_num_classes)
        self.fc_out_contlex = nn.ModuleDict(
            {
                str(pos_class): nn.Linear(embed_size, contlex_output_map[pos_class])
                for pos_class in contlex_output_map
            }
        )

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out_pos.weight)
        for pos_class in contlex_output_map:
            nn.init.xavier_uniform_(self.fc_out_contlex[str(pos_class)].weight)

        self.pos_weight = 1.0
        self.contlex_weight = 1.0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.contlex_output_map = contlex_output_map
        self.pos_class_sorted = list(sorted(contlex_output_map.keys()))
        self.contlex_class_max = max(contlex_output_map.values())

        self.pos_f1 = MulticlassF1Score(num_classes=pos_num_classes, average="weighted")
        self.contlex_f1 = MulticlassF1Score(
            num_classes=self.contlex_class_max, average="weighted"
        )

    def forward(self, x, pos_target=None):
        x = self.embedding(x)
        transformer_out = self.transformer_encoder(x)
        transformer_last_hidden = transformer_out[:, -1, :]

        pos_output = self.fc_out_pos(transformer_last_hidden)
        pos_labels = (
            pos_target if pos_target is not None else torch.argmax(pos_output, dim=1)
        )

        contlex_output = torch.zeros(
            (x.size(0), self.contlex_class_max), device=self.device
        )
        for pos_class in self.pos_class_sorted:
            contlex_size = self.hparams.contlex_output_map[pos_class]
            indices = (pos_labels == int(pos_class)).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                fc_out_contlex = self.fc_out_contlex[str(pos_class)]
                contlex_out = fc_out_contlex(transformer_last_hidden[indices])
                contlex_output[indices, :contlex_size] = contlex_out

        return pos_output, contlex_output

    def custom_loss(self, pos_output, contlex_output, pos_target, contlex_target):
        pos_loss = nn.CrossEntropyLoss()(pos_output, pos_target)
        contlex_loss = nn.CrossEntropyLoss()(contlex_output, contlex_target)
        return (
            self.pos_weight * pos_loss + self.contlex_weight * contlex_loss,
            pos_loss,
            contlex_loss,
        )

    def training_step(self, batch, batch_idx):
        x, pos_y, contlex_y = batch
        pos_output, contlex_output = self(x, pos_target=pos_y)
        total_loss, pos_loss, contlex_loss = self.custom_loss(
            pos_output, contlex_output, pos_y, contlex_y
        )
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, pos_y, contlex_y = batch
        pos_output, contlex_output = self(x, pos_target=pos_y)
        total_loss, pos_loss, contlex_loss = self.custom_loss(
            pos_output, contlex_output, pos_y, contlex_y
        )
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        pos_preds = torch.argmax(pos_output, dim=1)
        contlex_preds = torch.argmax(contlex_output, dim=1)
        pos_acc = (pos_preds == pos_y).float().mean()
        contlex_acc = (contlex_preds == contlex_y).float().mean()
        self.log("val_pos_acc", pos_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_contlex_acc", contlex_acc, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class SentencePieceProcessorWrapper:
    """
    Wrapper for SentencePiece tokenization using pre-trained models.
    """

    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def encode(self, text):
        """Encodes text using SentencePiece model."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        """Decodes token ids back into text using SentencePiece model."""
        return self.sp.decode(ids)


class ModelManager:

    def __init__(self, sentencepiece_model_file, encoder_path, model_checkpoint_path):

        self.tokenizer = SentencePieceProcessorWrapper(
            model_file=sentencepiece_model_file
        )

        # Load the encoders
        self.encoder_manager = LabelEncoderManager()
        self.encoder_manager.load_encoders(encoder_path)

        self.vocab_size = self.tokenizer.sp.get_piece_size()

        self.pos_num_classes = len(self.encoder_manager.pos_encoder.classes_)
        self.contlex_output_map = {
            pos_class: len(encoder.classes_)
            for pos_class, encoder in self.encoder_manager.contlex_encoders.items()
        }

        self.model = SharedEmbeddingTransformer.load_from_checkpoint(
            model_checkpoint_path
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, input_text, provided_pos=None):
        """
        Predict POS and Contlex labels for a given input text.

        Args:
            input_text: A single input sentence or text.
            provided_pos: An optional POS label provided by the user. If provided, this will be used to predict Contlex.

        Returns:
            pos_label: Predicted POS label (or provided POS label if given).
            contlex_label: Predicted Contlex label based on the POS label.
        """
        x_tokenized = self.tokenizer.encode(input_text)
        x_tensor = torch.tensor([x_tokenized], dtype=torch.long).to(self.device)

        if provided_pos is None:
            self.model.eval()
            with torch.no_grad():
                pos_output, _ = self.model(x_tensor)
            pos_pred = torch.argmax(pos_output, dim=1)
            pos_label = self.encoder_manager.inverse_transform_pos([pos_pred.item()])[0]
        else:
            pos_label = provided_pos
            pos_pred = torch.tensor(
                self.encoder_manager.transform_pos([provided_pos]), dtype=torch.long
            ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, contlex_output = self.model(x_tensor, pos_target=pos_pred)
        contlex_pred = torch.argmax(contlex_output, dim=1)

        contlex_label = self.encoder_manager.inverse_transform_contlex(
            [pos_label], [contlex_pred.item()]
        )[0]

        return pos_label, contlex_label

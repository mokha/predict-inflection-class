from model import ModelManager

model_manager = ModelManager(
    "./model/skolt_bpe.model", "./model/label_encoders.pkl", "./model/model.ckpt"
)

print(model_manager.predict("mii lea suomi", provided_pos="N"))

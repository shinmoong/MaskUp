from tensorflow import keras
model = keras.models.load_model("MaskModel_resize_32.h5", compile=False)

export_path = 'MaskModel_resize_32_pb'
model.save(export_path, save_format="tf")
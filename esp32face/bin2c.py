with open("face_mask_model.tflite", "rb") as f:
    data = f.read()

with open("face_mask_model.h", "w") as f:
    f.write("const unsigned char face_mask_model_tflite[] = {")
    f.write(",".join(map(lambda b: hex(b), data)))
    f.write("};\nconst int face_mask_model_tflite_len = " + str(len(data)) + ";")

from paddleocr import TextRecognition

model = TextRecognition(model_dir="./../../models/PP-OCRv5_mobile_rec_jp_fine_tuned_v4/complete_model",
                        model_name="PP-OCRv5_mobile_rec")

output = model.predict(input="./../../data/raw/images/recc.png", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="./../../output/v4")
    res.save_to_json(save_path="./../../output/v4")
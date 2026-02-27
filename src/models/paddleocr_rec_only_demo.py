from paddleocr import TextRecognition

model = TextRecognition(model_dir="./../../models/PP-OCRv5_mobile_rec_jp_fine_tuned/complete_model",
                        model_name="PP-OCRv5_mobile_rec")

output = model.predict(input="./../../data/processed/fine_tuning_answer_sheet/images/train_000003.jpg", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="./../../output/v1")
    res.save_to_json(save_path="./../../output/v1")
import paddlex
import os

p = paddlex.create_pipeline(pipeline="layout_parsing_config.yaml")
result = p.predict("./../../data/raw/images/shirobako.jpg")

out_dir = "./../../output/v4/layout_test2"
os.makedirs(out_dir, exist_ok=True)

for res in result:
    lr = res['layout_det_res']
    print(f"=== PP-DocLayout-L: {len(lr['boxes'])} boxes ===")
    for b in lr['boxes']:
        print(f"  label={b['label']}, score={b['score']:.3f}, coord=[{b['coordinate'][0]:.0f},{b['coordinate'][1]:.0f},{b['coordinate'][2]:.0f},{b['coordinate'][3]:.0f}]")

    res.save_to_img(out_dir)

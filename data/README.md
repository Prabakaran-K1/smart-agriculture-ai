# Crop recommendation data

`crop_recommendation.csv` is the 2,200-row crop recommendation dataset published in the [Harvestify repository](https://github.com/Gladiator07/Harvestify/tree/master/Data-processed/crop_recommendation.csv), which credits the [Kaggle crop recommendation dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

The dataset contains N, P, K, temperature, humidity, pH, rainfall, and crop labels for 22 crops. It is a demonstration dataset and is not locally calibrated or independently validated for a particular farm or region. Do not use its predictions as a substitute for local agronomic advice.

## Treatment planner

Treatment recommendations in `app.py` are transcribed from crop-specific Tamil Nadu Agricultural University (TNAU) crop-protection guidance:

- [Maize fall armyworm](https://agritech.tnau.ac.in/crop_protection/maize/crop_prot_maize_fall_armyworm.html), page updated May 2023.
- [Groundnut rust](https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d3.html), page footer dated 2015.
- [Groundnut Alternaria leaf disease](https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d6.html), page footer dated 2015.
- [Groundnut stem rot](https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d4.html), page footer dated 2015.
- [Tomato fruit borer](https://agritech.tnau.ac.in/crop_protection/tomato/tomato_1.html), page updated June 2023.
- [Tomato early blight](https://agritech.tnau.ac.in/crop_protection/pdf/2022/PSR%20August%202022.pdf), TNAU August 2022 surveillance report: Mancozeb 75% WP at 2 g/L, two applications at weekly intervals.
- [Tomato late blight](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html), TNAU page updated January 2023; rates are listed per acre.

Verify current registration for the exact crop and pest and follow the current product label before use. Some TNAU entries state spray concentration rather than a per-hectare rate; the planner asks for calibrated total spray-water volume instead of inventing one.

The maize INR pack-price ranges in the treatment catalogue are illustrative sample estimates, not live or independently verified dealer prices. The tomato early-blight Mancozeb pack estimate refers to this [Coimbatore supplier listing](https://www.indiamart.com/proddetail/powder-mancozeb-1kg-2852386355797.html), which listed about INR 250/kg; the UI marks it as an indicative, non-live offer. Other treatment options show that a local price quote is needed rather than fabricating prices. Check with a local retailer before purchase.

## Photo screening model

`static/models/plantvillage/` contains the MIT-licensed [ConCaPlant PlantVillage classifier](https://huggingface.co/imaflower/plantvillage-mobilenetv3) in ONNX format plus its class labels. The model has 15 labels covering tomato, potato, and bell pepper, and was evaluated on curated PlantVillage images. Field photos can produce incorrect results; the displayed model score is not field accuracy or a confirmed diagnosis. Only tomato early and late blight currently connect to the treatment catalogue. Other labels do not trigger medicine advice.

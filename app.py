from math import ceil, isfinite
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_recommendation.joblib"
API_FIELDS = ("nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall")
MODEL_FIELDS = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
VALUE_LIMITS = ((0, 200), (0, 200), (0, 200), (0, 50), (0, 100), (0, 14), (0, 300))
MAX_PHOTO_BYTES = 5 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_PHOTO_BYTES + 128 * 1024
SYMPTOM_ADVICE = {
    "yellow_leaves": {
        "label": "Yellowing leaves",
        "cause": "Nutrient imbalance, water stress, or natural leaf ageing",
        "action": "Check soil moisture and compare a soil test before changing fertilizer.",
    },
    "leaf_spots": {
        "label": "Leaf spots",
        "cause": "Possible fungal or bacterial leaf disease",
        "action": "Remove badly affected leaves, avoid wetting foliage, and ask a local specialist to identify the cause.",
    },
    "chewed_leaves": {
        "label": "Chewed leaves",
        "cause": "Possible chewing insects or other field pests",
        "action": "Inspect leaf undersides and nearby plants, identify the pest, and consider non-chemical controls first.",
    },
    "wilting": {
        "label": "Wilting",
        "cause": "Water stress, root damage, or waterlogged soil",
        "action": "Check root-zone moisture and drainage before changing irrigation.",
    },
    "white_powder": {
        "label": "White powder on leaves",
        "cause": "Possible powdery mildew or another surface growth",
        "action": "Improve airflow, avoid overhead watering, and have the plant checked before treatment.",
    },
    "stunted_growth": {
        "label": "Stunted growth",
        "cause": "Possible nutrient, pH, water, root, or pest stress",
        "action": "Review soil test results and inspect roots and nearby plants for a shared cause.",
    },
}
TREATMENT_CATALOG = {
    "maize_fall_armyworm": {
        "crop": "Maize",
        "problem": "Fall armyworm",
        "source": "https://agritech.tnau.ac.in/crop_protection/maize/crop_prot_maize_fall_armyworm.html",
        "source_name": "Tamil Nadu Agricultural University crop protection guidance",
        "source_updated": "May 2023",
        "stages": {
            "seed_treatment": {
                "name": "Seed treatment before sowing",
                "window": "Treat seed before sowing; use the current product label and calibrated seed quantity.",
                "products": [
                    {
                        "id": "maize_faw_seed_combo",
                        "name": "Cyantraniliprole 19.8% + Thiamethoxam 19.8% FS",
                        "formulation": "19.8% + 19.8% FS seed treatment",
                        "rate": 4,
                        "unit": "ml",
                        "rate_basis": "per_kg_seed",
                        "pack_size": 100,
                        "pack_unit": "ml",
                        "price_min": None,
                        "price_max": None,
                    },
                ],
            },
            "pre_sowing": {
                "name": "Land preparation and crop establishment",
                "window": "Apply neem cake during the last ploughing; establish pheromone traps for field monitoring.",
                "products": [
                    {"id": "maize_faw_neem_cake", "name": "Neem cake", "formulation": "Soil amendment", "rate": 250, "unit": "kg", "pack_size": 25, "pack_unit": "kg", "price_min": None, "price_max": None},
                    {"id": "maize_faw_pheromone_traps", "name": "Fall armyworm pheromone traps", "formulation": "Monitoring traps", "rate": 12, "unit": "traps", "pack_size": 1, "pack_unit": "trap", "price_min": None, "price_max": None},
                ],
            },
            "early_whorl": {
                "name": "Early whorl stage",
                "window": "15-20 days after emergence",
                "products": [
                    {
                        "id": "azadirachtin_1500",
                        "name": "Azadirachtin 1500 ppm",
                        "formulation": "1500 ppm",
                        "rate": 2500,
                        "unit": "ml",
                        "pack_size": 1000,
                        "pack_unit": "ml",
                        "price_min": 300,
                        "price_max": 700,
                    },
                    {
                        "id": "chlorantraniliprole_185",
                        "name": "Chlorantraniliprole 18.5 SC",
                        "formulation": "18.5% SC",
                        "rate": 200,
                        "unit": "ml",
                        "pack_size": 100,
                        "pack_unit": "ml",
                        "price_min": 900,
                        "price_max": 1600,
                    },
                    {
                        "id": "flubendiamide_480",
                        "name": "Flubendiamide 480 SC",
                        "formulation": "480 g/L SC",
                        "rate": 250,
                        "unit": "ml",
                        "pack_size": 100,
                        "pack_unit": "ml",
                        "price_min": 800,
                        "price_max": 1400,
                    },
                ],
            },
            "late_whorl": {
                "name": "Late whorl stage",
                "window": "35-40 days after emergence",
                "products": [
                    {
                        "id": "metarhizium_ma_gdu",
                        "name": "Metarhizium anisopliae (TNAU-MA-GDU isolate)",
                        "formulation": "TNAU-MA-GDU isolate",
                        "rate": 2.5,
                        "unit": "kg",
                        "pack_size": 1,
                        "pack_unit": "kg",
                        "price_min": 250,
                        "price_max": 600,
                    },
                    {
                        "id": "emamectin_5sg",
                        "name": "Emamectin benzoate 5 SG",
                        "formulation": "5% SG",
                        "rate": 200,
                        "unit": "g",
                        "pack_size": 100,
                        "pack_unit": "g",
                        "price_min": 180,
                        "price_max": 350,
                    },
                    {
                        "id": "novaluron_10ec",
                        "name": "Novaluron 10 EC",
                        "formulation": "10% EC",
                        "rate": 750,
                        "unit": "ml",
                        "pack_size": 500,
                        "pack_unit": "ml",
                        "price_min": 500,
                        "price_max": 900,
                    },
                    {
                        "id": "spinetoram_117sc",
                        "name": "Spinetoram 11.7 SC",
                        "formulation": "11.7% SC",
                        "rate": 250,
                        "unit": "ml",
                        "pack_size": 100,
                        "pack_unit": "ml",
                        "price_min": 850,
                        "price_max": 1500,
                    },
                ],
            },
            "cob_formation": {
                "name": "Cob formation, if needed",
                "window": "TNAU allows one late-whorl option at cob formation if needed; avoid automatic repetition.",
                "products": [
                    {"id": "maize_faw_cob_emamectin", "name": "Emamectin benzoate 5 SG", "formulation": "5% SG", "rate": 200, "unit": "g", "pack_size": 100, "pack_unit": "g", "price_min": 180, "price_max": 350},
                    {"id": "maize_faw_cob_novaluron", "name": "Novaluron 10 EC", "formulation": "10% EC", "rate": 750, "unit": "ml", "pack_size": 500, "pack_unit": "ml", "price_min": 500, "price_max": 900},
                    {"id": "maize_faw_cob_spinetoram", "name": "Spinetoram 11.7 SC", "formulation": "11.7% SC", "rate": 250, "unit": "ml", "pack_size": 100, "pack_unit": "ml", "price_min": 850, "price_max": 1500},
                ],
            },
        },
    },
    "groundnut_rust": {
        "crop": "Groundnut",
        "problem": "Rust",
        "source": "https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d3.html",
        "source_name": "Tamil Nadu Agricultural University groundnut rust guidance",
        "source_updated": "Page footer: 2015",
        "stages": {
            "foliar_disease": {
                "name": "Rust observed on leaves",
                "window": "Spray when rust is confirmed; TNAU says repeat after 15 days only if necessary.",
                "products": [
                    {"id": "gn_rust_mancozeb", "name": "Mancozeb", "formulation": "Fungicide", "rate": 1000, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_chlorothalonil", "name": "Chlorothalonil", "formulation": "Fungicide", "rate": 1000, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_sulphur", "name": "Wettable sulphur", "formulation": "Fungicide", "rate": 2500, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_tridemorph", "name": "Tridemorph", "formulation": "Fungicide", "rate": 500, "unit": "ml", "pack_size": 500, "pack_unit": "ml", "price_min": None, "price_max": None},
                ],
            },
            "repeat_if_needed": {
                "name": "Repeat only if needed",
                "window": "TNAU says repeat the foliar spray after 15 days only if necessary.",
                "products": [
                    {"id": "gn_rust_repeat_mancozeb", "name": "Mancozeb", "formulation": "Fungicide", "rate": 1000, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_repeat_chlorothalonil", "name": "Chlorothalonil", "formulation": "Fungicide", "rate": 1000, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_repeat_sulphur", "name": "Wettable sulphur", "formulation": "Fungicide", "rate": 2500, "unit": "g", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_rust_repeat_tridemorph", "name": "Tridemorph", "formulation": "Fungicide", "rate": 500, "unit": "ml", "pack_size": 500, "pack_unit": "ml", "price_min": None, "price_max": None},
                ],
            },
        },
    },
    "groundnut_alternaria": {
        "crop": "Groundnut",
        "problem": "Alternaria leaf disease",
        "source": "https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d6.html",
        "source_name": "Tamil Nadu Agricultural University groundnut disease guidance",
        "source_updated": "Page footer: 2015",
        "stages": {
            "foliar_disease": {
                "name": "Leaf disease observed",
                "window": "Foliar application; confirm the disease before spraying.",
                "products": [
                    {"id": "gn_alt_mancozeb", "name": "Mancozeb", "formulation": "0.3% spray solution", "rate": 3, "unit": "g", "rate_basis": "per_litre_spray", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_alt_copper", "name": "Copper oxychloride", "formulation": "0.3% spray solution", "rate": 3, "unit": "g", "rate_basis": "per_litre_spray", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "gn_alt_carbendazim", "name": "Carbendazim", "formulation": "0.1% spray solution", "rate": 1, "unit": "g", "rate_basis": "per_litre_spray", "pack_size": 500, "pack_unit": "g", "price_min": None, "price_max": None},
                ],
            },
        },
    },
    "groundnut_stem_rot": {
        "crop": "Groundnut",
        "problem": "Stem rot",
        "source": "https://agritech.tnau.ac.in/crop_protection/groundnut_diseases/groundnut_d4.html",
        "source_name": "Tamil Nadu Agricultural University groundnut disease guidance",
        "source_updated": "Page footer: 2015",
        "stages": {
            "seed_treatment": {
                "name": "Seed treatment before sowing",
                "window": "Treat seed before sowing at the rate in the TNAU guidance; verify current product registration and label.",
                "products": [
                    {"id": "gn_stemrot_seed_trichoderma", "name": "Trichoderma viride", "formulation": "Bio-control formulation", "rate": 4, "unit": "g", "rate_basis": "per_kg_seed", "pack_size": 100, "pack_unit": "g", "price_min": None, "price_max": None},
                ],
            },
            "soil_application": {
                "name": "Soil application with FYM",
                "window": "Apply with 50 kg well-decomposed FYM per hectare as described by TNAU; confirm application timing locally.",
                "products": [
                    {"id": "gn_stemrot_trichoderma", "name": "Trichoderma viride", "formulation": "Bio-control formulation", "rate": 2.5, "unit": "kg", "pack_size": 1, "pack_unit": "kg", "price_min": None, "price_max": None},
                ],
            },
        },
    },
    "tomato_fruit_borer": {
        "crop": "Tomato",
        "problem": "Fruit borer (Helicoverpa armigera)",
        "source": "https://agritech.tnau.ac.in/crop_protection/tomato/tomato_1.html",
        "source_name": "Tamil Nadu Agricultural University tomato crop-protection guidance",
        "source_updated": "June 2023",
        "stages": {
            "flowering_fruiting": {
                "name": "Flowering and fruiting",
                "window": "Trichogramma release begins at flower initiation and repeats at 7-day intervals as advised.",
                "products": [
                    {"id": "tomato_helilure_traps", "name": "Helilure pheromone traps", "formulation": "Monitoring traps", "rate": 12, "unit": "traps", "pack_size": 1, "pack_unit": "trap", "price_min": None, "price_max": None},
                    {"id": "tomato_trichogramma", "name": "Trichogramma pretiosum", "formulation": "Biological control; live beneficial insects", "rate": 100000, "unit": "parasitoids", "pack_size": 100000, "pack_unit": "parasitoids", "price_min": None, "price_max": None},
                    {"id": "tomato_hanpv", "name": "HaNPV", "formulation": "Biological control; verify product potency on its label", "rate": 1500000000000, "unit": "POBs", "pack_size": 1000000000000, "pack_unit": "POBs", "price_min": None, "price_max": None},
                ],
            },
            "field_establishment": {
                "name": "Crop establishment and monitoring",
                "window": "Install pheromone traps for monitoring; use scouting and local threshold guidance before any spray.",
                "products": [
                    {"id": "tomato_faw_establishment_traps", "name": "Helilure pheromone traps", "formulation": "Monitoring traps", "rate": 12, "unit": "traps", "rate_basis": "per_hectare", "pack_size": 1, "pack_unit": "trap", "price_min": None, "price_max": None},
                ],
            },
            "flower_initiation": {
                "name": "Flower initiation",
                "window": "Release Trichogramma at flower initiation; TNAU advises 7-day intervals.",
                "products": [
                    {"id": "tomato_faw_flower_trichogramma", "name": "Trichogramma pretiosum", "formulation": "Biological control; live beneficial insects", "rate": 100000, "unit": "parasitoids", "rate_basis": "per_hectare", "pack_size": 100000, "pack_unit": "parasitoids", "price_min": None, "price_max": None},
                ],
            },
            "fruit_borer_observed": {
                "name": "Eggs or young larvae found during scouting",
                "window": "TNAU lists HaNPV for egg/early-larval monitoring findings; verify product potency and label directions.",
                "products": [
                    {"id": "tomato_faw_scout_hanpv", "name": "HaNPV", "formulation": "Biological control; verify product potency on its label", "rate": 1500000000000, "unit": "POBs", "rate_basis": "per_hectare", "pack_size": 1000000000000, "pack_unit": "POBs", "price_min": None, "price_max": None},
                ],
            },
        },
    },
    "tomato_early_blight": {
        "crop": "Tomato",
        "problem": "Early blight (Alternaria solani)",
        "source": "https://agritech.tnau.ac.in/crop_protection/pdf/2022/PSR%20August%202022.pdf",
        "source_name": "TNAU Pest and Disease Surveillance and Forecast Report",
        "source_updated": "August 2022",
        "stages": {
            "foliar_disease": {
                "name": "Early blight observed on leaves",
                "window": "TNAU report advises two applications at weekly intervals after symptoms are noticed.",
                "products": [
                    {
                        "id": "tomato_early_mancozeb75",
                        "name": "Mancozeb 75% WP",
                        "formulation": "75% WP",
                        "rate": 2,
                        "unit": "g",
                        "rate_basis": "per_litre_spray",
                        "pack_size": 1000,
                        "pack_unit": "g",
                        "price_min": 250,
                        "price_max": 250,
                        "price_source_url": "https://www.indiamart.com/proddetail/powder-mancozeb-1kg-2852386355797.html",
                        "price_source_note": "A Coimbatore supplier listed generic Mancozeb powder at ₹250/kg; the exact 75% WP brand/pack price may differ and this is not a live local quote.",
                    },
                    {"id": "tomato_early_iprodione", "name": "Iprodione 50% WP", "formulation": "50% WP", "rate": 600, "unit": "g", "rate_basis": "per_acre", "pack_size": 500, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_azoxystrobin", "name": "Azoxystrobin 23% SC", "formulation": "23% SC", "rate": 200, "unit": "ml", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "ml", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_mancozeb35", "name": "Mancozeb 35% SC", "formulation": "35% SC", "rate": 1000, "unit": "g", "rate_basis": "per_acre", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_metiram", "name": "Metiram 70% WG", "formulation": "70% WG", "rate": 1000, "unit": "g", "rate_basis": "per_acre", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_pyraclostrobin", "name": "Pyraclostrobin 20% WG", "formulation": "20% WG", "rate": 200, "unit": "g", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_zineb", "name": "Zineb 75% WP", "formulation": "75% WP", "rate": 800, "unit": "g", "rate_basis": "per_acre", "pack_size": 500, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_azoxy_difen", "name": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC", "formulation": "18.2% + 11.4% SC", "rate": 200, "unit": "ml", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "ml", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_metiram_pyraclo", "name": "Metiram 55% + Pyraclostrobin 5% WG", "formulation": "55% + 5% WG", "rate": 600, "unit": "g", "rate_basis": "per_acre", "pack_size": 500, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                    {"id": "tomato_early_tebu_triflox", "name": "Tebuconazole 50% + Trifloxystrobin 25% WG", "formulation": "50% + 25% WG", "rate": 140, "unit": "g", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "g", "price_min": None, "price_max": None, "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html", "source_name": "TNAU tomato early blight integrated disease management", "source_updated": "June 2023"},
                ],
            },
            "weekly_followup": {
                "name": "Second spray, 7 days later",
                "window": "TNAU August 2022 report says to repeat Mancozeb 75% WP once at a weekly interval.",
                "products": [
                    {"id": "tomato_early_mancozeb75_followup", "name": "Mancozeb 75% WP", "formulation": "75% WP", "rate": 2, "unit": "g", "rate_basis": "per_litre_spray", "pack_size": 1000, "pack_unit": "g", "price_min": 250, "price_max": 250, "price_source_url": "https://www.indiamart.com/proddetail/powder-mancozeb-1kg-2852386355797.html", "price_source_note": "Indicative generic Mancozeb powder supplier listing; exact 75% WP pack price can differ."},
                ],
            },
        },
    },
    "tomato_late_blight": {
        "crop": "Tomato",
        "problem": "Late blight (Phytophthora infestans)",
        "source": "https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html",
        "source_name": "TNAU tomato late blight integrated disease management",
        "source_updated": "January 2023",
        "stages": {
            "foliar_disease": {
                "name": "Late blight observed on leaves or stems",
                "window": "TNAU recommends integrated disease management; follow the current product label.",
                "products": [
                    {"id": "tomato_late_cyazofamid", "name": "Cyazofamid 34.5% SC", "formulation": "34.5% SC", "rate": 80, "unit": "ml", "rate_basis": "per_acre", "pack_size": 80, "pack_unit": "ml", "price_min": 160, "price_max": 160, "price_source_url": "https://www.indiamart.com/proddetail/ranman-cyazofamid-34-5-sc-fungicide-27405279097.html", "price_source_note": "IndiaMART seller lists ₹2,000/litre and an 80 ml pack; ₹160/pack is a calculated indicative price, not a local quote."},
                    {"id": "tomato_late_azoxy", "name": "Azoxystrobin 23% SC", "formulation": "23% SC", "rate": 200, "unit": "ml", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "ml", "price_min": None, "price_max": None},
                    {"id": "tomato_late_mancozeb", "name": "Mancozeb 35% SC", "formulation": "35% SC", "rate": 1000, "unit": "g", "rate_basis": "per_acre", "pack_size": 1000, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "tomato_late_zineb", "name": "Zineb 75% WP", "formulation": "75% WP", "rate": 800, "unit": "g", "rate_basis": "per_acre", "pack_size": 500, "pack_unit": "g", "price_min": None, "price_max": None},
                    {"id": "tomato_late_azoxy_difen", "name": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC", "formulation": "18.2% + 11.4% SC", "rate": 200, "unit": "ml", "rate_basis": "per_acre", "pack_size": 100, "pack_unit": "ml", "price_min": None, "price_max": None},
                    {"id": "tomato_late_trichoderma", "name": "Trichoderma asperellum", "formulation": "Bio-control formulation; soil application", "rate": 1, "unit": "kg", "rate_basis": "per_acre", "pack_size": 1, "pack_unit": "kg", "price_min": None, "price_max": None},
                ],
            },
        },
    },
}


def train_crop_model():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Crop training data not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    required_columns = {*MODEL_FIELDS, "label"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Crop dataset is missing columns: {', '.join(sorted(missing_columns))}")

    dataset = dataset.dropna(subset=[*MODEL_FIELDS, "label"])
    features = dataset.loc[:, MODEL_FIELDS].apply(pd.to_numeric, errors="coerce")
    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows]
    labels = dataset.loc[valid_rows, "label"].astype(str)
    if labels.nunique() < 2:
        raise ValueError("Crop dataset must contain at least two crop labels.")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    validation_accuracy = accuracy_score(test_labels, model.predict(test_features))

    profile_data = features.copy()
    profile_data["label"] = labels
    crop_profiles = profile_data.groupby("label")[["N", "P", "K", "rainfall"]].mean()
    crop_profiles = {
        str(crop): {field: round(float(value), 1) for field, value in profile.items()}
        for crop, profile in crop_profiles.to_dict(orient="index").items()
    }

    artifact = {
        "model": model,
        "validation_accuracy": validation_accuracy,
        "training_rows": len(features),
        "crop_count": labels.nunique(),
        "crop_profiles": crop_profiles,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def load_crop_model():
    if MODEL_PATH.is_file():
        artifact = joblib.load(MODEL_PATH)
        if "crop_profiles" in artifact:
            return artifact
    return train_crop_model()


MODEL_ARTIFACT = load_crop_model()
MODEL = MODEL_ARTIFACT["model"]


def build_fertilizer_advice(values, crop):
    profile = MODEL_ARTIFACT["crop_profiles"][crop]
    nutrient_fields = (("N", "nitrogen"), ("P", "phosphorus"), ("K", "potassium"))
    nutrient_advice = []
    for nutrient, field in nutrient_fields:
        current = values[API_FIELDS.index(field)]
        reference = profile[nutrient]
        tolerance = max(reference * 0.15, 2)
        if current < reference - tolerance:
            status = "below"
            action = "Below the dataset average; confirm with a soil test before adding nutrients."
        elif current > reference + tolerance:
            status = "above"
            action = "Above the dataset average; avoid extra application without soil-test guidance."
        else:
            status = "near"
            action = "Near the dataset average; base any application on a current soil test."
        nutrient_advice.append({
            "nutrient": nutrient,
            "value": round(current, 1),
            "reference": reference,
            "status": status,
            "action": action,
        })

    return {
        "summary": f"Your NPK readings are compared with the training-data average for {crop}.",
        "items": nutrient_advice,
        "note": "These comparisons are not fertilizer rates. Confirm soil conditions locally before applying fertilizer.",
    }


def build_irrigation_advice(values, crop):
    rainfall = values[API_FIELDS.index("rainfall")]
    reference = MODEL_ARTIFACT["crop_profiles"][crop]["rainfall"]
    if rainfall < reference * 0.85:
        status = "Below crop reference"
        guidance = "Check root-zone moisture regularly; provide supplemental water only when the soil and crop need it."
    elif rainfall > reference * 1.2:
        status = "Above crop reference"
        guidance = "Monitor drainage and standing water; protect roots from waterlogging."
    else:
        status = "Near crop reference"
        guidance = "Keep monitoring soil moisture and adjust watering to field conditions."

    return {
        "status": status,
        "rainfall": round(rainfall, 1),
        "reference": reference,
        "guidance": guidance,
        "note": "Seasonal rainfall comparison only; this is not an irrigation schedule or water-volume estimate.",
    }


def build_plant_health_advice(symptoms):
    observations = [SYMPTOM_ADVICE[symptom] for symptom in symptoms]
    if not observations:
        summary = "No symptoms selected. Scout the crop regularly for changes."
    else:
        summary = f"First checks for {len(observations)} selected observation(s)."
    return {
        "summary": summary,
        "observations": observations,
        "note": "These are general checks, not a diagnosis. Confirm the cause with a local crop specialist before treatment.",
    }


def build_assessment(values, symptoms):
    model_input = pd.DataFrame([values], columns=MODEL_FIELDS)
    probabilities = MODEL.predict_proba(model_input)[0]
    top_indices = probabilities.argsort()[::-1][:3]
    crops = [
        {"name": str(MODEL.classes_[index]), "fit": round(float(probabilities[index]) * 100)}
        for index in top_indices
    ]
    crops[0]["reason"] = "This crop received the highest probability from the trained Random Forest model."
    recommended_crop = crops[0]["name"]
    return {
        "crops": crops,
        "advisories": {
            "fertilizer": build_fertilizer_advice(values, recommended_crop),
            "irrigation": build_irrigation_advice(values, recommended_crop),
            "plantHealth": build_plant_health_advice(symptoms),
        },
        "model": {
            "type": "Random Forest",
            "trainingRows": MODEL_ARTIFACT["training_rows"],
            "cropCount": MODEL_ARTIFACT["crop_count"],
            "validationAccuracy": round(MODEL_ARTIFACT["validation_accuracy"] * 100, 1),
        },
    }


def supported_image_type(contents):
    if contents.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(contents) >= 12 and contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
        return "image/webp"
    return None

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send field conditions as JSON."), 400

    try:
        values = [float(data[name]) for name in API_FIELDS]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Enter a valid number for every field condition."), 400

    if any(
        not isfinite(value) or value < lower or value > upper
        for value, (lower, upper) in zip(values, VALUE_LIMITS)
    ):
        return jsonify(error="One or more values are outside the accepted range."), 400

    symptoms = data.get("symptoms", [])
    if not isinstance(symptoms, list) or any(symptom not in SYMPTOM_ADVICE for symptom in symptoms):
        return jsonify(error="Choose only symptoms from the provided list."), 400

    return jsonify(build_assessment(values, symptoms))


@app.route("/analyze-photo", methods=["POST"])
def analyze_photo():
    photo = request.files.get("photo")
    if photo is None:
        return jsonify(error="Choose a crop photo to upload."), 400

    image_bytes = photo.stream.read(MAX_PHOTO_BYTES + 1)
    if len(image_bytes) > MAX_PHOTO_BYTES:
        return jsonify(error="Photos must be 5 MB or smaller."), 413
    image_type = supported_image_type(image_bytes)
    if image_type is None:
        return jsonify(error="Use a JPEG, PNG, or WebP image."), 400

    try:
        values = [float(request.form[name]) for name in API_FIELDS]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Enter a valid number for every field condition."), 400

    if any(
        not isfinite(value) or value < lower or value > upper
        for value, (lower, upper) in zip(values, VALUE_LIMITS)
    ):
        return jsonify(error="One or more values are outside the accepted range."), 400

    symptoms = request.form.getlist("symptoms")
    if any(symptom not in SYMPTOM_ADVICE for symptom in symptoms):
        return jsonify(error="Choose only symptoms from the provided list."), 400

    assessment = build_assessment(values, symptoms)
    assessment["photo"] = {
        "contentType": image_type,
        "sizeBytes": len(image_bytes),
        "stored": False,
        "notice": "Photo received, but no image-recognition model is configured. Crop-health advice is based on the symptoms you selected, not visual diagnosis.",
    }
    return jsonify(assessment)


@app.route("/treatment-cases", methods=["GET"])
def treatment_cases():
    return jsonify(cases=[
        {
            "id": case_id,
            "crop": treatment_case["crop"],
            "problem": treatment_case["problem"],
            "stages": [
                {"id": stage_id, "name": stage["name"], "window": stage["window"]}
                for stage_id, stage in treatment_case["stages"].items()
            ],
        }
        for case_id, treatment_case in TREATMENT_CATALOG.items()
    ])


@app.route("/treatment-options", methods=["GET"])
def treatment_options():
    case_id = request.args.get("case_id", "")
    stage_id = request.args.get("stage", "")
    treatment_case = TREATMENT_CATALOG.get(case_id)
    stage = treatment_case["stages"].get(stage_id) if treatment_case else None
    if stage is None:
        return jsonify(error="Choose a supported crop, confirmed pest, and crop stage."), 400

    return jsonify(
        crop=treatment_case["crop"],
        problem=treatment_case["problem"],
        stage=stage["name"],
        window=stage["window"],
        source=treatment_case["source"],
        sourceName=treatment_case["source_name"],
        sourceNote=treatment_case["source_updated"],
        products=[
            {
                "id": product["id"],
                "name": product["name"],
                "formulation": product["formulation"],
                "rate": product["rate"],
                "unit": product["unit"],
                "rateBasis": product.get("rate_basis", "per_hectare"),
                "samplePriceMinPerPack": product["price_min"],
                "samplePriceMaxPerPack": product["price_max"],
                "packSize": product["pack_size"],
                "packUnit": product["pack_unit"],
                "priceSourceUrl": product.get("price_source_url"),
                "priceSourceNote": product.get("price_source_note"),
            }
            for product in stage["products"]
        ],
    )


@app.route("/treatment-plan", methods=["POST"])
def treatment_plan():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send the treatment selection as JSON."), 400

    treatment_case = TREATMENT_CATALOG.get(data.get("case_id", ""))
    stage = treatment_case["stages"].get(data.get("stage", "")) if treatment_case else None
    if stage is None:
        return jsonify(error="Choose a supported crop, confirmed pest, and crop stage."), 400

    product = next((item for item in stage["products"] if item["id"] == data.get("product_id")), None)
    if product is None:
        return jsonify(error="Choose a treatment listed for this crop stage."), 400

    rate_basis = product.get("rate_basis", "per_hectare")
    if rate_basis == "per_litre_spray":
        try:
            spray_volume_litres = float(data["spray_volume_litres"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter the calibrated total spray-water volume in litres."), 400
        if not isfinite(spray_volume_litres) or not 0 < spray_volume_litres <= 1000000:
            return jsonify(error="Spray-water volume must be greater than 0 and no more than 1,000,000 litres."), 400
        area_hectares = None
        required_quantity = product["rate"] * spray_volume_litres
    elif rate_basis == "per_kg_seed":
        try:
            seed_quantity_kg = float(data["seed_quantity_kg"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter the seed quantity to be treated in kilograms."), 400
        if not isfinite(seed_quantity_kg) or not 0 < seed_quantity_kg <= 1000000:
            return jsonify(error="Seed quantity must be greater than 0 and no more than 1,000,000 kg."), 400
        area_hectares = None
        area_acres = None
        spray_volume_litres = None
        required_quantity = product["rate"] * seed_quantity_kg
    else:
        try:
            area_hectares = float(data["area_hectares"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter the affected field area in hectares."), 400
        if not isfinite(area_hectares) or not 0 < area_hectares <= 10000:
            return jsonify(error="Field area must be greater than 0 and no more than 10,000 hectares."), 400
        spray_volume_litres = None
        if rate_basis == "per_acre":
            area_acres = area_hectares * 2.4710538147
            required_quantity = product["rate"] * area_acres
        else:
            area_acres = None
            required_quantity = product["rate"] * area_hectares

    packages_needed = ceil(required_quantity / product["pack_size"])
    purchase_quantity = packages_needed * product["pack_size"]
    if product["price_min"] is None or product["price_max"] is None:
        sample_low = None
        sample_high = None
        price_basis = "No verified sample price available; request a current local retailer quote."
    else:
        sample_low = packages_needed * product["price_min"]
        sample_high = packages_needed * product["price_max"]
        price_basis = "Indicative sample price per retail pack, not a live local quote."

    safety_notes = [
        f"Use only after confirming {treatment_case['problem']} on {treatment_case['crop']}; symptoms or a photo alone are not a diagnosis.",
        "The listed treatments are alternatives. Choose one suitable option only; do not tank-mix or repeat unless the current product label or a qualified local adviser says to.",
        f"Source note: {treatment_case['source_updated']}. Verify current registration and the exact crop, pest, application rate, pre-harvest interval, and PPE on the product label before use.",
        "Keep people, animals, water sources, and beneficial organisms protected as required by the product label.",
    ]
    if treatment_case["problem"] == "Stem rot":
        safety_notes.append("The cited recommendation is a biological soil treatment mixed with FYM, not a foliar spray.")
    if treatment_case["problem"] == "Fruit borer (Helicoverpa armigera)":
        safety_notes.append("Pheromone traps and beneficial-insect releases are monitoring/IPM tools, not pesticide sprays; follow the supplier's handling and release instructions.")

    return jsonify(
        crop=treatment_case["crop"],
        problem=treatment_case["problem"],
        stage=stage["name"],
        applicationWindow=stage["window"],
        product={
            "name": product["name"],
            "formulation": product["formulation"],
            "rate": product["rate"],
            "unit": product["unit"],
            "rateBasis": rate_basis,
        },
        quantity={
            "areaHectares": area_hectares,
            "areaAcres": round(area_acres, 3) if rate_basis == "per_acre" else None,
            "sprayVolumeLitres": spray_volume_litres,
            "seedQuantityKg": seed_quantity_kg if rate_basis == "per_kg_seed" else None,
            "required": round(required_quantity, 2),
            "unit": product["unit"],
            "packagesNeeded": packages_needed,
            "packageSize": product["pack_size"],
            "packageUnit": product["pack_unit"],
            "purchaseQuantity": round(purchase_quantity, 2),
        },
        cost={
            "currency": "INR",
            "sampleLow": sample_low,
            "sampleHigh": sample_high,
            "priceMinPerPack": product["price_min"],
            "priceMaxPerPack": product["price_max"],
            "packSize": product["pack_size"],
            "packUnit": product["pack_unit"],
            "basis": price_basis,
            "priceSourceUrl": product.get("price_source_url"),
            "priceSourceNote": product.get("price_source_note"),
        },
        source={
            "name": treatment_case["source_name"],
            "url": treatment_case["source"],
            "note": treatment_case["source_updated"],
        },
        safetyNotes=safety_notes,
    )


@app.errorhandler(413)
def photo_too_large(_error):
    return jsonify(error="The upload is too large. Photos must be 5 MB or smaller."), 413

if __name__ == "__main__":
    app.run(debug=True)
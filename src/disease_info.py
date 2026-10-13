"""Agronomic knowledge base containing symptoms, causes, and remedies for Potato leaf diseases."""

DISEASE_DETAILS = {
    "Potato___Early_blight": {
        "common_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate to High",
        "symptoms": [
            "Small, dark brown to black spots with concentric rings ('target-board' appearance) on older lower leaves.",
            "Yellow halo (chlorotic ring) often surrounding the concentric lesions.",
            "Premature defoliation, leading to sunscald on tubers and reduced crop yield.",
        ],
        "favorable_conditions": [
            "Warm temperatures (24°C - 29°C / 75°F - 85°F).",
            "Alternating wet and dry weather cycles with prolonged leaf wetness (dew or sprinkler irrigation).",
        ],
        "organic_control": [
            "Prune and destroy infected lower leaves immediately; do not compost them.",
            "Apply copper-based fungicides (Bordeaux mixture) or bio-fungicides with Bacillus subtilis.",
            "Use neem oil spray or baking soda (potassium bicarbonate) spray as early preventive measures.",
        ],
        "chemical_control": [
            "Apply protectant fungicides like Mancozeb or Chlorothalonil early in the season.",
            "Use systemic fungicides such as Azoxystrobin, Difenoconazole, or Boscalid if infection spreads.",
        ],
        "prevention": [
            "Maintain a 3- to 4-year crop rotation with non-solanaceous crops (avoid tomatoes and eggplants).",
            "Use drip irrigation instead of overhead watering to keep foliage dry.",
            "Ensure adequate spacing between plants to maximize airflow and sunlight penetration.",
        ],
    },
    "Potato___Late_blight": {
        "common_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete / Water Mold)",
        "severity": "Critical / Severe",
        "symptoms": [
            "Irregular, pale to dark water-soaked lesions expanding rapidly on leaf tips and margins.",
            "White, velvety fungal growth on the underside of leaves during humid mornings.",
            "Rapid leaf collapse, blackening, and foul odor in dense foliage canopies.",
        ],
        "favorable_conditions": [
            "Cool to moderate temperatures (15°C - 21°C / 60°F - 70°F).",
            "High relative humidity (> 90%) and frequent rainfall or prolonged morning fog.",
        ],
        "organic_control": [
            "Uproot and burn heavily infected plants immediately to prevent neighborhood spore spread.",
            "Apply preventive Copper hydroxide or Copper oxychloride before wet weather windows.",
            "Spray compost tea or Trichoderma viride formulations to support phyllosphere resistance.",
        ],
        "chemical_control": [
            "Apply systemic translaminar fungicides such as Metalaxyl / Mefenoxam combined with Mancozeb.",
            "Use Cymoxanil, Dimethomorph, or Fluopicolide when high risk of active sporulation is forecasted.",
        ],
        "prevention": [
            "Plant certified disease-free certified seed tubers.",
            "Hill up soil around potato hills adequately to protect tubers from surface spore wash-off.",
            "Destroy all volunteer potato plants and cull piles before the growing season begins.",
        ],
    },
    "Potato___healthy": {
        "common_name": "Healthy Plant",
        "pathogen": "None (Plant is in optimal physiological health)",
        "severity": "None",
        "symptoms": [
            "Uniform vibrant green coloration across leaf surfaces.",
            "Intact leaf margins and veins with no concentric spots, water-soaking, or necrotic lesions.",
            "Strong turgidity and active vegetative growth.",
        ],
        "favorable_conditions": [
            "Balanced soil moisture, adequate sunlight (6-8 hours daily), and well-draining nutrient-rich soil.",
        ],
        "organic_control": [
            "Continue routine organic feeding with well-composted organic matter and seaweed extract.",
            "Encourage beneficial predatory insects (ladybugs, lacewings) to prevent pest vectors.",
        ],
        "chemical_control": [
            "No chemical fungicides or interventions required.",
        ],
        "prevention": [
            "Continue balanced N-P-K fertilization (avoid excessive nitrogen which promotes soft foliar growth).",
            "Maintain regular weekly field scouting to catch early signs of pests or disease spores.",
        ],
    },
}


import io
import json
import base64
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 1. LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "disease_efficientnet.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

KNOWLEDGE_PATH = os.path.join(
    BASE_DIR,
    "disease_knowledge.json"
)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 2. LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_data = json.load(f)

KNOWLEDGE_PATH = os.path.join(BASE_DIR, "disease_knowledge.json")

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    disease_knowledge = json.load(f)

print("Disease knowledge base loaded successfully.")
print("Number of disease classes:", len(disease_knowledge))


if isinstance(class_data, list):
    class_names = class_data

elif isinstance(class_data, dict):
    class_names = [
        class_data[str(i)]
        for i in range(len(class_data))
    ]

else:
    raise ValueError("Unsupported class_names.json format")


print("Number of classes:", len(class_names))


# ============================================================
# 3. GET TRAINED MODEL LAYERS
# ============================================================

base_model = model.get_layer("efficientnetb0")

augmentation = model.get_layer(
    "data_augmentation"
)

gap_layer = model.get_layer(
    "global_average_pooling2d"
)

dropout_layer = model.get_layer(
    "dropout"
)

classifier_layer = model.get_layer(
    "disease_prediction"
)


# ============================================================
# 4. GRAD-CAM FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("top_conv").output
)

print("Feature extractor model created successfully.")


# ============================================================
# 5. GRAD-CAM FUNCTION
# ============================================================

def make_gradcam_heatmap(
    image,
    pred_index=None
):

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    with tf.GradientTape() as tape:

        augmented = augmentation(
            image_batch,
            training=False
        )

        conv_outputs = feature_extractor(
            augmented,
            training=False
        )

        x = gap_layer(conv_outputs)

        x = dropout_layer(
            x,
            training=False
        )

        predictions = classifier_layer(x)

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / tf.maximum(
        tf.reduce_max(heatmap),
        tf.keras.backend.epsilon()
    )

    return (
        heatmap.numpy(),
        int(pred_index)
    )


# ============================================================
# 6. DISEASE INFORMATION
# ============================================================

disease_information = {

    "Apple___Apple_scab": {
        "reason":
            "Apple scab is caused by a fungal pathogen and is favored by conditions that allow moisture to remain on leaves.",
        "advice":
            "Remove heavily affected plant material and maintain good airflow around the crop."
    },

    "Tomato___Early_blight": {
        "reason":
            "Early blight is a fungal disease that commonly develops under favorable moisture and temperature conditions.",
        "advice":
            "Remove severely affected leaves, improve airflow and avoid prolonged leaf wetness."
    },

    "Tomato___Late_blight": {
        "reason":
            "Late blight is caused by a pathogen that can spread rapidly under cool and humid conditions.",
        "advice":
            "Remove affected plant material and maintain good field sanitation."
    },

    "Potato___Early_blight": {
        "reason":
            "Early blight in potato is associated with fungal infection and can become more severe when plants are stressed.",
        "advice":
            "Remove severely affected foliage and maintain appropriate crop management."
    },

    "Potato___Late_blight": {
        "reason":
            "Late blight is a destructive potato disease that spreads readily under favorable cool and humid conditions.",
        "advice":
            "Remove affected material and monitor surrounding plants carefully."
    }

}


# ============================================================
# 7. GENERIC INFORMATION
# ============================================================

def get_disease_information(
    disease_name
):

    if disease_name in disease_information:

        return disease_information[
            disease_name
        ]

    return {
        "reason":
            "The model has identified visual patterns associated with this disease class. Environmental conditions, crop stress and pathogen presence can contribute to disease development.",
        "advice":
            "Inspect the affected plant carefully and monitor nearby plants for similar symptoms."
    }


# ============================================================
# 8. CONVERT HEATMAP TO IMAGE
# ============================================================

def create_gradcam_image(
    original_image,
    heatmap
):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Convert original image to numpy
    original = np.array(
        original_image
    ).astype(np.uint8)

    # Resize heatmap
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap_array = np.array(
        heatmap_image
    ) / 255.0

    # Apply colormap
    colored_heatmap = cm.jet(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    # Overlay
    overlay = (
        0.55 * original
        + 0.45 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    output_image = Image.fromarray(
        overlay
    )

    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# 9. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="KrishiSaarthi AI API",
    description="Explainable Crop Disease Detection API"
)


# ============================================================
# 10. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# 11. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "KrishiSaarthi AI backend is running."
    }


# ============================================================
# 12. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image_resized = image.resize(
        (224, 224)
    )


    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    image_array = np.array(
        image_resized
    ).astype(np.float32)


    image_tensor = tf.convert_to_tensor(
        image_array
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        tf.expand_dims(
            image_tensor,
            axis=0
        ),
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_index]
    )


    disease = class_names[
        predicted_index
    ]
    
    knowledge = disease_knowledge.get(
    disease,
    {
        "disease": disease,
        "crop": "Unknown",
        "cause": "Information not available.",
        "symptoms": [],
        "favorable_conditions": [],
        "precautions": [],
        "management": [],
        "prevention": []
    }
)


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap, _ = make_gradcam_heatmap(
        image_tensor,
        pred_index=predicted_index
    )


    gradcam_image = create_gradcam_image(
        image_resized,
        heatmap
    )


    # --------------------------------------------------------
    # Disease information
    # --------------------------------------------------------

    info = get_disease_information(
        disease
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "disease": disease,

        "confidence": confidence,

        "gradcam": gradcam_image,

        "reason": info["reason"],

        "advice": info["advice"],
        
        "knowledge": knowledge

    }
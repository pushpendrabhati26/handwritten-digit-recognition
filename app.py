
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw one large digit, then click Predict.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

canvas = st_canvas(
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    update_streamlit=True,
    key="digit_canvas",
    return_image_data=True,
)

if st.button("Predict Digit", type="primary"):

    if canvas.image_data is None:
        st.warning("Please draw a digit first.")

    else:
        # Convert the canvas to grayscale
        image = Image.fromarray(
            canvas.image_data.astype(np.uint8)
        ).convert("L")

        # Keep white strokes and remove very faint pixels
        image = image.point(lambda p: p if p > 30 else 0)

        # Find the drawn digit
        bbox = image.getbbox()

        if bbox is None:
            st.warning("Please draw a digit first.")

        else:
            image = image.crop(bbox)

            # Fit the digit inside a 20x20 area
            image.thumbnail(
                (20, 20),
                Image.Resampling.LANCZOS
            )

            # Place it on the MNIST-sized black canvas
            processed = Image.new("L", (28, 28), 0)

            x = (28 - image.width) // 2
            y = (28 - image.height) // 2

            processed.paste(image, (x, y))

            # Match the CNN's input shape and normalization
            array = np.asarray(
                processed, dtype=np.float32
            ) / 255.0

            array = array.reshape(1, 28, 28, 1)

            probabilities = model.predict(
                array, verbose=0
            )[0]

            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit]) * 100

            st.success(f"Predicted digit: {digit}")
            st.write(
                f"Model confidence: {confidence:.2f}%"
            )

            st.image(
                processed.resize(
                    (224, 224),
                    Image.Resampling.NEAREST
                ),
                caption="Processed 28 × 28 image sent to the model"
            )

            st.subheader("Scores for each digit")
            st.bar_chart({
                str(i): float(probabilities[i])
                for i in range(10)
            })


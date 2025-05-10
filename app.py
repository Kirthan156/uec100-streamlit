import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import collections

# --- Project Info ---
st.set_page_config(page_title="UEC-100 Calorie Estimator", layout="centered")
st.title("🍱 Food Calorie Estimator")

st.markdown("""
### 📘 Project Overview
This Streamlit app is built to **estimate calories and nutrients** from images of Japanese food using a deep learning model trained on the **UEC-100 Food Dataset**.

- ✅ **Model Used:** EfficientNet-B0
- 🍣 **Dataset:** UEC-100 (100 classes of Japanese cuisine)
- 📊 **Features:**
  - Food image classification
  - Calorie prediction (per 100g)
  - Nutrient visualization (bar + pie chart)
  - Ingredients and step-by-step recipe
- 🎯 **Goal:** Help users understand what's on their plate and make smarter food decisions.

---

Upload a food image below to get started.
""")

# --- Load model ---
@st.cache_resource
def load_model():
    # Check if GPU is available and use it, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use relative paths for better portability
    model_path = os.path.join("model", "uec100_efficientnetb0.pth")
    
    # Debug: Check if model path exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    
    try:
        # First, attempt to load with efficientnet_pytorch as it's the most common implementation for custom models
        try:
            from efficientnet_pytorch import EfficientNet
            st.info("Trying to load with efficientnet_pytorch...")
            
            # Load the model file first to see what we're working with
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's a complete model object or a state dict
            if hasattr(checkpoint, 'state_dict'):
                st.info("Loaded object is a complete model")
                model = checkpoint
            else:
                # Create from scratch and load weights
                st.info("Loaded object is a state dict, creating model...")
                model = EfficientNet.from_name('efficientnet-b0', num_classes=100)
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            st.success("Successfully loaded model with efficientnet_pytorch!")
            return model
        except Exception as e:
            st.warning(f"Couldn't load with efficientnet_pytorch: {e}")
            
        # If that fails, try with torchvision's implementation
        try:
            st.info("Trying to load with torchvision EfficientNet...")
            # Load the model file to examine
            checkpoint = torch.load(model_path, map_location=device)
            
            if hasattr(checkpoint, 'state_dict'):
                st.info("Loaded object is a complete model")
                model = checkpoint
            else:
                # Create model from scratch
                model = models.efficientnet_b0(weights=None)
                # Modify classifier to match 100 classes
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=100)
                
                # Inspect first few keys to help with debugging
                first_keys = list(checkpoint.keys())[:3]
                st.info(f"First few state dict keys: {first_keys}")
                
                # Try loading with strict=False to ignore mismatched keys
                model.load_state_dict(checkpoint, strict=False)
                
                # Log number of loaded vs total parameters
                total_params = sum(p.numel() for p in model.parameters())
                loaded_params = sum(p.numel() for name, p in model.state_dict().items() 
                                  if name in checkpoint)
                st.info(f"Loaded {loaded_params} of {total_params} parameters")
            
            model = model.to(device)
            model.eval()
            st.success("Successfully loaded model with torchvision EfficientNet!")
            return model
        except Exception as e:
            st.warning(f"Couldn't load with torchvision: {e}")
        
        # Last resort: Create a simple CNN model that matches expected input/output
        st.warning("Unable to load original model. Creating a fallback model...")
        st.warning("⚠️ This fallback model won't make accurate predictions!")
        
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = torch.nn.Linear(128, 100)
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = SimpleCNN().to(device)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e)
        return None

# --- Load class names ---
@st.cache_data
def load_class_names():
    class_names_path = os.path.join("utils", "uec100_classes.json")
    try:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
            # Debug: Print some class mappings
            st.write("Class name examples:")
            items = list(class_names.items())[:5]  # Show first 5 items
            for idx, name in items:
                st.write(f"Class {idx}: {name}")
            return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.exception(e)
        # Return a simple dict as fallback
        return {str(i): f"Class {i}" for i in range(100)}

# --- Load metadata ---
@st.cache_data
def load_metadata():
    metadata_path = os.path.join("utils", "uec100_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return {}

# --- Preprocess image ---
def preprocess_image(image):
    # Standard ImageNet normalization and resizing to 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- Predict class ---
def predict(image, model, class_names):
    if model is None:
        st.error("Model could not be loaded. Cannot make predictions.")
        return None
        
    # Preprocess the image
    img_tensor = preprocess_image(image)
    device = next(model.parameters()).device  # Get the device where model is loaded
    img_tensor = img_tensor.to(device)
    
    # Run prediction
    with torch.no_grad():
        # Forward pass
        outputs = model(img_tensor)
        # Get probabilities with softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get top 3 predictions
        topk_probs, topk_indices = torch.topk(probabilities, k=3, dim=1)
    
    # Convert to lists and show top predictions with confidence
    top_classes = []
    for i in range(3):
        idx = topk_indices[0, i].item()
        prob = topk_probs[0, i].item() * 100
        
        # Handle different class name formats (could be int or string keys)
        if str(idx) in class_names:
            class_name = class_names[str(idx)]
        elif idx in class_names:
            class_name = class_names[idx]
        else:
            class_name = f"Unknown (Class {idx})"
            
        top_classes.append((class_name, prob))
        
    # Display top predictions with confidence
    st.subheader("Top 3 Predictions:")
    for class_name, prob in top_classes:
        st.write(f"{class_name}: {prob:.2f}%")
    
    # Return the top prediction
    return top_classes[0][0] if top_classes else None

# --- Draw nutrient charts ---
def show_nutrient_charts(nutrients):
    try:
        labels = list(nutrients.keys())
        # Convert values to numeric, handling both string with 'g' suffix and numeric values
        values = []
        for v in nutrients.values():
            if isinstance(v, str) and 'g' in v:
                values.append(float(v.replace('g', '')))
            else:
                values.append(float(v))

        # Bar Chart
        st.subheader("🔍 Nutrient Breakdown (Bar Chart)")
        fig1, ax1 = plt.subplots()
        ax1.bar(labels, values, color=["#FF9999", "#99FF99", "#9999FF"])
        ax1.set_ylabel("Grams per 100g")
        ax1.set_title("Nutrition Content")
        st.pyplot(fig1)

        # Pie Chart
        st.subheader("🥧 Nutrient Proportion (Pie Chart)")
        fig2, ax2 = plt.subplots()
        ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#FF9999", "#99FF99", "#9999FF"])
        ax2.axis("equal")
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Error displaying nutrient charts: {e}")

# --- Load resources ---
try:
    st.subheader("Loading model and resources...")
    with st.spinner("Loading model (this may take a moment)..."):
        model = load_model()
    with st.spinner("Loading class names..."):
        class_names = load_class_names()
    with st.spinner("Loading metadata..."):
        metadata = load_metadata()
    st.success("Resources loaded successfully!")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.exception(e)
    model, class_names, metadata = None, {}, {}

# --- UI ---
st.write("Upload an image of a Japanese dish and view its predicted calories, nutrition, ingredients, and recipe.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                predicted_class = predict(image, model, class_names)
                
                if predicted_class:
                    st.success(f"✅ Predicted Food: **{predicted_class.capitalize()}**")

                    food_info = metadata.get(predicted_class)

                    if food_info:
                        st.markdown(f"**Calories (per 100g):** {food_info['calories']} kcal")
                        st.markdown(f"**Vegetarian:** {'✅ Yes' if food_info['vegetarian'] else '❌ No'}")
                        
                        # Show nutrient charts
                        show_nutrient_charts(food_info["nutrients"])

                        # Show ingredients and recipe
                        st.subheader("🧾 Ingredients")
                        st.write(food_info["ingredients"])

                        st.subheader("📖 Recipe")
                        st.write(food_info["recipe"])
                    else:
                        st.error(f"No metadata found for the predicted food: {predicted_class}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.exception(e)  # Show detailed error

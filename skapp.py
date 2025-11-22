import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Set page configuration (icon and title)
st.set_page_config(page_title="Bird Species Classifier", page_icon="🦜", layout="wide")

# Add custom styles to make the app visually appealing
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        h1 {
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            background-color: #F0F0F0;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.header("🦜 Bird Species Classifier")
st.sidebar.markdown("Upload a bird image to classify its species using a trained CNN model.")

# Load the trained model
MODEL_PATH = 'C:/Users/kadus/Downloads/bird_species_classifier (1).h5'
model = load_model(MODEL_PATH)

# List of bird species labels (this needs to correspond to your model's output classes)
class_labels = ['Acadian_Flycatcher', 'American_Crow', 'American_Goldfinch',
       'American_Pipit', 'American_Redstart',
       'American_Three_toed_Woodpecker', 'Anna_Hummingbird', 'Artic_Tern',
       'Baird_Sparrow', 'Baltimore_Oriole', 'Bank_Swallow',
       'Barn_Swallow', 'Bay_breasted_Warbler', 'Belted_Kingfisher',
       'Bewick_Wren', 'Black_Tern', 'Black_and_white_Warbler',
       'Black_billed_Cuckoo', 'Black_capped_Vireo',
       'Black_footed_Albatross', 'Black_throated_Blue_Warbler',
       'Black_throated_Sparrow', 'Blue_Grosbeak', 'Blue_Jay',
       'Blue_headed_Vireo', 'Blue_winged_Warbler', 'Boat_tailed_Grackle',
       'Bobolink', 'Bohemian_Waxwing', 'Brandt_Cormorant',
       'Brewer_Blackbird', 'Brewer_Sparrow', 'Bronzed_Cowbird',
       'Brown_Creeper', 'Brown_Pelican', 'Brown_Thrasher', 'Cactus_Wren',
       'California_Gull', 'Canada_Warbler', 'Cape_Glossy_Starling',
       'Cape_May_Warbler', 'Cardinal', 'Carolina_Wren', 'Caspian_Tern',
       'Cedar_Waxwing', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
       'Chipping_Sparrow', 'Chuck_will_Widow', 'Clark_Nutcracker',
       'Clay_colored_Sparrow', 'Cliff_Swallow', 'Common_Raven',
       'Common_Tern', 'Common_Yellowthroat', 'Crested_Auklet',
       'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eared_Grebe',
       'Eastern_Towhee', 'Elegant_Tern', 'European_Goldfinch',
       'Evening_Grosbeak', 'Field_Sparrow', 'Fish_Crow', 'Florida_Jay',
       'Forsters_Tern', 'Fox_Sparrow', 'Frigatebird', 'Gadwall',
       'Geococcyx', 'Glaucous_winged_Gull', 'Golden_winged_Warbler',
       'Grasshopper_Sparrow', 'Gray_Catbird', 'Gray_Kingbird',
       'Gray_crowned_Rosy_Finch', 'Great_Crested_Flycatcher',
       'Great_Grey_Shrike', 'Green_Jay', 'Green_Kingfisher',
       'Green_Violetear', 'Green_tailed_Towhee', 'Groove_billed_Ani',
       'Harris_Sparrow', 'Heermann_Gull', 'Henslow_Sparrow',
       'Herring_Gull', 'Hooded_Merganser', 'Hooded_Oriole',
       'Hooded_Warbler', 'Horned_Grebe', 'Horned_Lark', 'Horned_Puffin',
       'House_Sparrow', 'House_Wren', 'Indigo_Bunting', 'Ivory_Gull',
       'Kentucky_Warbler', 'Laysan_Albatross', 'Lazuli_Bunting',
       'Le_Conte_Sparrow', 'Least_Auklet', 'Least_Flycatcher',
       'Least_Tern', 'Lincoln_Sparrow', 'Loggerhead_Shrike',
       'Long_tailed_Jaeger', 'Louisiana_Waterthrush', 'Magnolia_Warbler',
       'Mallard', 'Mangrove_Cuckoo', 'Marsh_Wren', 'Mockingbird',
       'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler',
       'Nelson_Sharp_tailed_Sparrow', 'Nighthawk', 'Northern_Flicker',
       'Northern_Fulmar', 'Northern_Waterthrush',
       'Olive_sided_Flycatcher', 'Orange_crowned_Warbler',
       'Orchard_Oriole', 'Ovenbird', 'Pacific_Loon', 'Painted_Bunting',
       'Palm_Warbler', 'Parakeet_Auklet', 'Pelagic_Cormorant',
       'Philadelphia_Vireo', 'Pied_Kingfisher', 'Pied_billed_Grebe',
       'Pigeon_Guillemot', 'Pileated_Woodpecker', 'Pine_Grosbeak',
       'Pine_Warbler', 'Pomarine_Jaeger', 'Prairie_Warbler',
       'Prothonotary_Warbler', 'Purple_Finch', 'Red_bellied_Woodpecker',
       'Red_breasted_Merganser', 'Red_cockaded_Woodpecker',
       'Red_eyed_Vireo', 'Red_faced_Cormorant', 'Red_headed_Woodpecker',
       'Red_legged_Kittiwake', 'Red_winged_Blackbird',
       'Rhinoceros_Auklet', 'Ring_billed_Gull', 'Ringed_Kingfisher',
       'Rock_Wren', 'Rose_breasted_Grosbeak', 'Ruby_throated_Hummingbird',
       'Rufous_Hummingbird', 'Rusty_Blackbird', 'Sage_Thrasher',
       'Savannah_Sparrow', 'Sayornis', 'Scarlet_Tanager',
       'Scissor_tailed_Flycatcher', 'Scott_Oriole', 'Seaside_Sparrow',
       'Shiny_Cowbird', 'Slaty_backed_Gull', 'Song_Sparrow',
       'Sooty_Albatross', 'Spotted_Catbird', 'Summer_Tanager',
       'Swainson_Warbler', 'Tennessee_Warbler', 'Tree_Sparrow',
       'Tree_Swallow', 'Tropical_Kingbird', 'Vermilion_Flycatcher',
       'Vesper_Sparrow', 'Warbling_Vireo', 'Western_Grebe',
       'Western_Gull', 'Western_Meadowlark', 'Western_Wood_Pewee',
       'Whip_poor_Will', 'White_Pelican', 'White_breasted_Kingfisher',
       'White_breasted_Nuthatch', 'White_crowned_Sparrow',
       'White_eyed_Vireo', 'White_necked_Raven', 'White_throated_Sparrow',
       'Wilson_Warbler', 'Winter_Wren', 'Worm_eating_Warbler',
       'Yellow_Warbler', 'Yellow_bellied_Flycatcher',
       'Yellow_billed_Cuckoo', 'Yellow_breasted_Chat',
       'Yellow_headed_Blackbird', 'Yellow_throated_Vireo']   # Add all species names here

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Reshape to match model input (1, 224, 224, 3)
    return image / 255.0  # Normalize the image

# Main layout using columns
st.title("🦜 Bird Species Classifier")
st.write("Upload an image of a bird to classify its species.")

# Columns layout for image upload and prediction result side by side
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a bird image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    # Preprocess the image and predict using the model
    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_class]
    
    # Display the prediction result with a large and styled font
    with col2:
        st.success(f"### Predicted Bird Species: **{predicted_label}** 🦜")
    
    # Option to classify another image
    st.markdown("---")
    if st.button("Classify Another Image"):
        st.experimental_rerun()

# Sidebar with additional information about the app and model
st.sidebar.markdown("""
    ## About
    This app uses a Convolutional Neural Network (CNN) model to classify bird species based on uploaded images.
    
    ### Instructions:
    1. Upload a JPG or PNG image of a bird.
    2. The app will display the uploaded image and classify the species.
    3. View the predicted bird species on the right.
""")

# Footer section for additional information
st.markdown("""
    <footer style="text-align:center; margin-top:50px;">
        <p>Created by <strong>Your Name</strong> | Bird Species Classifier | 2024</p>
    </footer>
""", unsafe_allow_html=True)

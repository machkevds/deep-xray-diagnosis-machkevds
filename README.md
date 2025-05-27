# Pneumonia Detection from Chest X-rays (with Grad-CAM)

This project uses a MobileNetV2-based convolutional neural network to detect pneumonia from chest X-ray images. It includes a web demo powered by Streamlit and interpretable Grad-CAM visualizations.

---

## Run the App in Google Colab

You can run the Streamlit app directly from Google Colab:

## 1. Upload the following to your Colab session:
- `xrayShowcaseApp.py` — the Streamlit app.
- `pneumonia_mobilenetv2_kvpch10.keras` — the trained model.

## 2. Run the following setup:
### - Install dependencies
!pip install streamlit

### - Fetch tunnel IP for deployment
!wget -q -O - ipv4.icanhazip.com

## - Launch the Streamlit app and create public link
!streamlit run xrayShowcaseApp.py & npx localtunnel --port 8501
'''

## EXTRA NOTES:
### - When prompted, you may see the following, type y and hit enter.

Need to install the following packages:
localtunnel@2.0.2
Ok to proceed? (y)

### - You will be asked to confirm tunnel IP with a Prompt
E.g. 'XX.XXX.XX.XX'


### - The link will be printed in the output cell. Example:
#### Your app is live at: 'https://silent-example-go.loca.lt'

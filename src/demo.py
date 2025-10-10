import gradio as gr 

convert_img = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  
    transforms.ToTensor()
])

network.eval()

def pass_image(image):
    image_tensor = convert_img(image).unsqueeze(0)  
    with torch.no_grad():
        result = network(image_tensor)
        pred = result.argmax(dim=1).item()

    class_names = torchvision.datasets.FashionMNIST.classes
    return class_names[pred]


demo = gr.Interface(pass_image, gr.Image(type="pil", image_mode="L", width=400, height=400), gr.Textbox())
demo.launch(share=True)

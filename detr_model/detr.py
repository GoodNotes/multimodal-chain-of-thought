from PIL import Image
import torchvision.transforms as T

from detr_model.model import get_pretrained_detr_embedder


class DETRModel:
    def __init__(self):
        self.model = get_pretrained_detr_embedder()
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_features(self, fp):
        im = Image.open(fp)
        img = self.transform(im).unsqueeze(0)
        return self.model(img).squeeze()


if __name__ == '__main__':
    model = DETRModel()
    _image_path = '/Users/hugochu/PycharmProjects/multimodal-chain-of-thought/data/raw_images/train/1/image.png'
    f = model.get_features(_image_path)
    print(f)
    print(f.shape)

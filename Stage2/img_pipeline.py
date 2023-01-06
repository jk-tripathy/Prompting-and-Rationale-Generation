import numpy as np
import torch
import clip
import torchvision.models as models
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from facenet_pytorch import MTCNN

import warnings
warnings.filterwarnings("ignore")

class ImagePipelines:
    def __init__(self) -> None:
        self.tasks = ['image type', 'object detection', 'indoor scene', 'outdoor scene', 'face']
        # cuda device
        device = 0 if torch.cuda.is_available() else -1

        # huggging face models
        # object detection
        self.object_model_id = 'facebook/detr-resnet-101'
        self.object_feature_extractor = AutoFeatureExtractor.from_pretrained(self.object_model_id)
        self.object_model = AutoModelForObjectDetection.from_pretrained(self.object_model_id)
        self.object_pipe = pipeline(
            'object-detection',
            model = self.object_model,
            feature_extractor = self.object_feature_extractor,
            device = device,
        )

        # indoor scene
        self.indoor_model_id = 'vincentclaes/mit-indoor-scenes'
        self.indoor_feature_extractor = AutoFeatureExtractor.from_pretrained(self.indoor_model_id)
        self.indoor_model = AutoModelForImageClassification.from_pretrained(self.indoor_model_id)
        self.indoor_pipe = pipeline(
            'image-classification',
            model = self.indoor_model,
            feature_extractor = self.indoor_feature_extractor,
            device = device,
        )

        # facial emotion
        self.facial_emotion_model_id = 'Rajaram1996/FacialEmoRecog'
        self.facial_emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(self.facial_emotion_model_id)
        self.facial_emotion_model = AutoModelForImageClassification.from_pretrained(self.facial_emotion_model_id)
        self.emotion_pipe = pipeline(
            'image-classification',
            model = self.facial_emotion_model,
            feature_extractor = self.facial_emotion_feature_extractor,
            device = device,
        )

        # gender 
        self.gender_model_id = "rizvandwiki/gender-classification-2"
        self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(self.gender_model_id)
        self.gender_model = AutoModelForImageClassification.from_pretrained(self.gender_model_id)
        self.gender_pipe = pipeline(
            'image-classification',
            model = self.gender_model,
            feature_extractor = self.gender_feature_extractor,
            device = device,
        )

        # age 
        self.age_model_id = "nateraw/vit-age-classifier"
        self.age_feature_extractor = AutoFeatureExtractor.from_pretrained(self.age_model_id)
        self.age_model = AutoModelForImageClassification.from_pretrained(self.age_model_id)
        self.age_pipe = pipeline(
            'image-classification',
            model = self.age_model,
            feature_extractor = self.age_feature_extractor,
            device = device,
        )

        # places365 - outdoor scene

        # load the pre-trained weights
        model_file = 'resnet50_places365.pth.tar'
        self.places365_model = models.__dict__['resnet50'](num_classes=365)
        checkpoint = torch.load(model_file, map_location= torch.device(f'cuda:{device}'))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.places365_model.load_state_dict(state_dict)
        self.places365_model.eval()

        self.places365_centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        self.places365_classes = tuple(classes)

        # clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

        # mtcnn - face detection
        self.mtcnn = MTCNN(device = device)

    def outdoor_pipe(self, image_path):
        # load the test image
        img = Image.open(image_path)
        input_img = V(self.places365_centre_crop(img).unsqueeze(0))

        # forward pass
        logit = self.places365_model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the prediction
        results = []
        for i in range(0, 5):
            results.append({'score': probs[i].item(), 'label': self.places365_classes[idx[i]]})

        return results

    def image_type_pipe(self, image_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_types = ["This is a image", "This is a sketch", "This is a cartoon", "This is a painting"]
        text = clip.tokenize(image_types).to(device)

        with torch.no_grad():
            logits_per_image, _ = self.clip_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        result = [{'score': float(score), 'label': label} for score, label in zip(probs[0], image_types)]

        return result

    def face_pipe(self, image_path):
        result = []
        img = Image.open(image_path)
        bb, score = self.mtcnn.detect(img)
        if bb is not None and score is not None:
            for score, bb in zip(score, bb):
                cropped_img = img.crop(bb)
                emotion = self.emotion_pipe(cropped_img)
                gender = self.gender_pipe(cropped_img)
                age = self.age_pipe(cropped_img)
                result.append({
                    'detection': {
                        'score': float(score),
                        'box': {
                            'xmin': float(bb[0]),
                            'ymin': float(bb[1]),
                            'xmax': float(bb[2]),
                            'ymax': float(bb[3]),
                        },
                    },
                    'age': age,
                    'emotion': emotion,
                    'gender': gender,
                })
        return result

    def get_results(self, image_path, task_label):
        if task_label == 'object detection':
            return self.object_pipe(image_path)
        elif task_label == 'indoor scene':
            return self.indoor_pipe(image_path)
        elif task_label == 'outdoor scene':
            return self.outdoor_pipe(image_path)
        elif task_label == 'face':
            return self.face_pipe(image_path)
        elif task_label == 'image type':
            return self.image_type_pipe(image_path)


if __name__ == '__main__':
    image_path = '../data/MAMI/images/28.jpg'
    img = Image.open(image_path)
    image_models = ImagePipelines()
    result = image_models.get_results(image_path, 'face')
    print(result)

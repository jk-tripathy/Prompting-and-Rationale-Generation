from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection, \
    AutoModelForVision2Seq, AutoTokenizer
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import clip
from facenet_pytorch import MTCNN
import time

import warnings 
warnings.filterwarnings("ignore")

class ImagePipelines:
    def __init__(self):
        print('Loading models')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        torch.cuda.empty_cache()

        self.tasks = ['image type', 'object detection', 'indoor scene', 'outdoor scene']
        # huggging face models
        # object detection
        self.object_model_id = 'facebook/detr-resnet-101'
        self.object_feature_extractor = AutoFeatureExtractor.from_pretrained(self.object_model_id)
        self.object_model = AutoModelForObjectDetection.from_pretrained(self.object_model_id)
        self.object_pipe = pipeline(
            'object-detection',
            model=self.object_model,
            feature_extractor=self.object_feature_extractor, device=0
        )


        # indoor scene
        self.indoor_model_id = 'vincentclaes/mit-indoor-scenes'
        self.indoor_feature_extractor = AutoFeatureExtractor.from_pretrained(self.indoor_model_id)
        self.indoor_model = AutoModelForImageClassification.from_pretrained(self.indoor_model_id)
        self.indoor_pipe = pipeline(
            'image-classification',
            model=self.indoor_model,
            feature_extractor=self.indoor_feature_extractor, device=0
        )

        # facial emotion
        self.facial_emotion_model_id = 'Rajaram1996/FacialEmoRecog'
        self.facial_emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(self.facial_emotion_model_id)
        self.facial_emotion_model = AutoModelForImageClassification.from_pretrained(self.facial_emotion_model_id)
        self.emotion_pipe = pipeline(
            'image-classification',
            model=self.facial_emotion_model,
            feature_extractor=self.facial_emotion_feature_extractor, device=0
        )

        self.mtcnn = MTCNN(keep_all=True)
        # self.mtcnn.to(self.device)

        # places365 - outdoor scene
        # load the pre-trained weights
        model_file = 'Stage2/resnet50_places365.pth.tar'
        self.places365_model = models.__dict__['resnet50'](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.places365_model.load_state_dict(state_dict)
        self.places365_model.eval()
        # self.places365_model.to(self.device)

        self.places365_centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'Stage2/categories_places365.txt'
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        self.places365_classes = tuple(classes)

        # clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        print('Done')

    def outdoor_pipe(self, image):
        # load the test image

        input_img = V(self.places365_centre_crop(image).unsqueeze(0))

        # forward pass
        logit = self.places365_model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the prediction
        results = []
        for i in range(0, 10):
            results.append({'score': probs[i].item(), 'label': self.places365_classes[idx[i]]})

        return results

    def image_type_pipe(self, image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = self.clip_preprocess(image).unsqueeze(0).to(device)
        image_types = ["This is an image", "This is a sketch", "This is a cartoon", "This is a painting"]
        text = clip.tokenize(image_types).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.clip_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        result = [{'score': float(score), 'label': label} for score, label in zip(probs[0], image_types)]
        return result

    def face_pipe(self, image):
        result = []
        bb, score = self.mtcnn.detect(image)
        if bb is not None and score is not None:
            for score, bb in zip(score, bb):
                cropped_img = image.crop(bb)
                emotion = self.emotion_pipe(cropped_img)
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
                    'emotion': emotion,
                })
        return result


    def get_results(self, image_path):
        visual_tags = {}

        image = Image.open(image_path)

        try:
            visual_tags['object_detection']= self.object_pipe(image)
        except:
            visual_tags['object_detection'] = ''

        try:
            visual_tags['indoor_scene'] = self.indoor_pipe(image)
        except:
            visual_tags['indoor_scene'] = ''

        try:
            visual_tags['places'] = self.outdoor_pipe(image)
        except:
            visual_tags['places'] = ''

        try:
            visual_tags['image_type'] = self.image_type_pipe(image)
        except:
            visual_tags['image_type'] = ''

        try:
            visual_tags['face'] = self.face_pipe(image)
        except:
            visual_tags['face'] = ''

        return visual_tags


if __name__ == '__main__':
    image_path = 'data/MAMI/images/28.jpg'
    img = Image.open(image_path)
    image_models = ImagePipelines()
    result = image_models.get_results(image_path)
    print(result)

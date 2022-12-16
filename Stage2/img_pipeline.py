import sys
sys.path.insert(0, "../")
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection, AutoModelForVision2Seq, AutoTokenizer
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import clip

import warnings
warnings.filterwarnings("ignore")

#from PIL import Image
#import torch
#from torchvision import transforms
#from transformers import OFATokenizer, OFAModel
#from OFA.models import sequence_generator
#from modelscope.pipelines import pipeline
#from modelscope.utils.constant import Tasks
#from modelscope.outputs import OutputKeys

class ImagePipelines:
    def __init__(self) -> None:
        self.tasks = ['image type', 'object detection', 'indoor scene', 'outdoor scene', 'facial emotion', 'caption']
        #huggging face models
        # object detection
        self.object_model_id = 'facebook/detr-resnet-101'
        self.object_feature_extractor = AutoFeatureExtractor.from_pretrained(self.object_model_id)
        self.object_model = AutoModelForObjectDetection.from_pretrained(self.object_model_id)
        self.object_pipe = pipeline(
            'object-detection',
            model = self.object_model,
            feature_extractor = self.object_feature_extractor,
        )

        #indoor scene
        self.indoor_model_id = 'vincentclaes/mit-indoor-scenes'
        self.indoor_feature_extractor = AutoFeatureExtractor.from_pretrained(self.indoor_model_id)
        self.indoor_model = AutoModelForImageClassification.from_pretrained(self.indoor_model_id)
        self.indoor_pipe = pipeline(
            'image-classification',
            model = self.indoor_model,
            feature_extractor = self.indoor_feature_extractor,
        )

        #facial emotion
        self.facial_emotion_model_id = 'Rajaram1996/FacialEmoRecog'
        self.facial_emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(self.facial_emotion_model_id)
        self.facial_emotion_model = AutoModelForImageClassification.from_pretrained(self.facial_emotion_model_id)
        self.emotion_pipe = pipeline(
            'image-classification',
            model = self.facial_emotion_model,
            feature_extractor = self.facial_emotion_feature_extractor,
        )

        #caption
        self.caption_model_id = 'nlpconnect/vit-gpt2-image-captioning'
        self.caption_feature_extractor = AutoFeatureExtractor.from_pretrained(self.caption_model_id)
        self.caption_model = AutoModelForVision2Seq.from_pretrained(self.caption_model_id)
        self.caption_tokenizer = AutoTokenizer.from_pretrained(self.caption_model_id)
        self.caption_pipe = pipeline(
            'image-to-text',
            model = self.caption_model,
            feature_extractor = self.caption_feature_extractor,
            tokenizer = self.caption_tokenizer,
        )

        #places365 - outdoor scene
        arch = 'resnet50'

        # load the pre-trained weights
        model_file = 'places365/%s_places365.pth.tar' % arch
        self.places365_model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.places365_model.load_state_dict(state_dict)
        self.places365_model.eval()

        self.places365_centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'places365/categories_places365.txt'
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        self.places365_classes = tuple(classes)

        #clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

    def outdoor_pipe(self, image_path):
        # load the test image
        img = Image.open(image_path)
        input_img = V(self.places365_centre_crop(img).unsqueeze(0))

        # forward pass
        logit = self.places365_model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the prediction
        results = {'output' : []}
        for i in range(0, 5):
            results['output'].append({'score': probs[i].item(), 'label': self.places365_classes[idx[i]]})

        return results
    
    def image_type_pipe(self, image_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_types = ["This is a image", "This is a sketch", "This is a cartoon", "This is a painting"] 
        text = clip.tokenize(image_types).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.clip_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
        result = {'output': [{'score': float(score), 'label': label} for score, label in zip(probs[0], image_types)]}
        return result
        
    def get_results(self, image_path, task_label): 
        if task_label == 'object detection':
            return self.object_pipe(image_path)
        elif task_label == 'indoor scene':
            return self.indoor_pipe(image_path)
        elif task_label == 'outdoor scene':
            return self.outdoor_pipe(image_path)
        elif task_label == 'facial emotion':
            return self.emotion_pipe(image_path)
        elif task_label == 'caption':
            return self.caption_pipe(image_path)
        elif task_label == 'image type':
            return self.image_type_pipe(image_path)

if __name__=='__main__':
    path_to_image = '../data/MAMI/images/28.jpg'
    image_models = ImagePipelines()
    result = image_models.get_results(path_to_image, task_label='caption') 
    print(result)

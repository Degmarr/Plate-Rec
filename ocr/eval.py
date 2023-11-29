import cv2 as cv
import torchvision.transforms as transforms
from PIL import Image
from ocr.model import Model


import torch
import torch.utils.data
import torch.nn.functional as F
from ocr.utils import AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 4
saved_model = 'ocr/modules/best_accuracy_5999_98.8011988011988.pth'
imgH = 32
imgW = 100
batch_max_length = 25
rgb = True
character = '0123456789abcdefghijklmnopqrstuvwxyz'
PAD = True
converter = AttnLabelConverter(character)
num_class = len(converter.character)
input_channel = 3

model = Model()
model = torch.nn.DataParallel(model).to(device)

print('loading pretrained model from %s' % saved_model)
model.load_state_dict(torch.load(saved_model, map_location=device))

imgH = 32
imgW = 100


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def AlignCollate(image):
    transform = ResizeNormalize((imgW, imgH))
    image_tensor = transform(image)
    image_tensor = torch.cat([image_tensor.unsqueeze(0)], 0)
    return image_tensor


def demo(im):
    img = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im_pil = im_pil.convert('L')

    image = AlignCollate(im_pil)

    # predict
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * 1).to(device)
        text_for_pred = torch.LongTensor(
            1, batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)[0]
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        pred_EOS = preds_str.find('[s]')
        # prune after "end of sentence" token ([s])
        pred = preds_str[:pred_EOS]
        pred_max_prob = preds_max_prob[0][:pred_EOS]
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    return pred, pred_max_prob

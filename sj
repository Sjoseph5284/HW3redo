
%%Problem 1 

eta=1;
update=.001;
epoch=5000.*(1e-2)
while update==.001
for i=1:4
y=p(i,:)*w';
if y>=0 & d(i)==0
w=w-eta*p(i,:)
up(i)=.001
end
end
nou=up*up %%number of updates to training model
if nou>0
update=1;
else update=0;
end
end

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = weathermodel.load_data()

Epochs = 5000

Activation function = sigmoid

Optimizer = sgd

model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='sigmoid'),
tf.keras.layers.Dense(10)
])
model.compile(optimizer='sgd',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)


%%Problem 2

%%resnet18 and densenet121 for nvidia trt pose


from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
WIDTH = 256
HEIGHT = 256
camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30, capture_device=1)
def rename_images(dir_):
    overall_count = 0
    #dir_ = dir_+dataset_name
    for i in range(len(os.listdir(dir_))):
        dir_to_check = os.path.join(dir_,"%s" % (i+1))
        dir_to_check+='/'
        for count, filename in enumerate(os.listdir(dir_to_check)):
            dst = "%08d.jpg"% overall_count
            src = dir_to_check+filename
            dst = dir_to_check+dst 
            os.rename(src, dst)
            overall_count+=1
            
 %%Problem 3
 config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
 def detect_image(img):
   
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
   
img_path = "images/blueangels.jpg"
prev_time = time.time()
img = Image.open(img_path)
detections = detect_image(img)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print ('Inference Time: %s' % (inference_time))
# Get bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
img = np.array(img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)

    unique_labels = detections[:, -1].cpu().unique()

plt.show()
 



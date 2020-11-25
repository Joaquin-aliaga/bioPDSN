import torch
from easydict import EasyDict as edict
from lib.Learner import face_learner
from lib.mtcnn import MTCNN
from torchvision import transforms as trans

#mtcnn = MTCNN()
print("mtcnn loaded")

conf = edict()
#conf.data_path = Path('data')
#conf.work_path = Path('work_space/')
#conf.model_path = conf.work_path/'models'
#conf.log_path = conf.work_path/'log'
#conf.save_path = conf.work_path/'save'
conf.input_size = [112,112]
conf.embedding_size = 512
conf.use_mobilfacenet = False
conf.net_depth = 50
conf.drop_ratio = 0.6
conf.net_mode = 'ir_se' # or 'ir'
conf.threshold = 0.5
conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",conf.device)
conf.test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
#conf.data_mode = 'emore'

learner = face_learner(conf,inference=True)
model.load_state_dict(torch.load("./weights/model_ir_se50.pth"))
model.eval()
print("Modelo cargado correctamente !! wuju")


# Face Recognition
import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)
    # face Not Found empty 0 return
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rests, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    # np.zeros = shape를 통해서 크기를 지정해주고, dtype으로 data type을 지정해준다.
     #그러면 shape와 dtype에 해당하는 0으로 채워진 배열을 반환해주는 함수
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rests.append(rect)
        shape = sp(img, d)

        # convert dlib shape to numpy array 
        for i in range(0, 68):
             shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
    return rests, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))
    return np.array(face_descriptors)

# Compute Saved Face Description
img_paths = {
    '소담': 'images/sodam.jpg',
    '이한': 'images/yi_han.jpg',
    '세종': 'images/sejong.jpg',
    '지희': 'images/jihee.jpg'
}

descs = {
    '소담': None,
    '이한': None,
    '세종': None,
    '지희': None
}

for name, img_paths in img_paths.items():
    img_bgr = cv2.imread(img_paths)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    _, img_shapes, _ = find_faces(img_rgb)

    descs[name] = encode_faces(img_rgb, img_shapes)[0]
    np.save('images/yuhee_team.npy', descs)
print(descs)
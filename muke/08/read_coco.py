from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

anno_file = "/home/kuan/MUKE-SOURCES/DATA/coco/annotations/instances_val2017.json"
coco = COCO(anno_file)

catIds = coco.getCatIds(catNms=['person'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)

for i in range(len(imgIds)):
    image = coco.loadImgs(imgIds[i])[0]
    I = io.imread(image["coco_url"])
    plt.imshow(I)
    anno_id = coco.getAnnIds(imgIds=image["id"], catIds=catIds, iscrowd=None)
    annotation = coco.loadAnns(anno_id)
    coco.showAnns(annotation)
    plt.show()
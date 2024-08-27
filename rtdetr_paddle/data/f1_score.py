import json

def calcular_iou_coco(bbox1, bbox2):
    """Calcula el índice de superposición de área (IoU) entre dos bounding boxes en formato COCO."""
    # Coordenadas y dimensiones de los bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Coordenadas de los puntos de esquina
    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1
    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2

    # Coordenadas de la intersección (esquina superior izquierda)
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    
    # Coordenadas de la intersección (esquina inferior derecha)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    
    # Área de la intersección
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Área de los bounding boxes
    area_bb1 = w1 * h1
    area_bb2 = w2 * h2
    
    # Área de la unión
    union_area = area_bb1 + area_bb2 - intersection_area
    
    # Calcular el IoU
    iou = intersection_area / union_area

    return iou

#with open('../bbox.json') as f:
with open('./dataset/gun/test_dextre/_annotations.coco.json') as f:
    d = json.load(f)
    images = d['images']
    print(len(images))
    anno = d['annotations']
    print(len(anno))

with open('./bbox.json') as f:
    d = json.load(f)
    n = len(d)//300
    cont = 0
    detection_gt = []
    detection_pr = []
    TP = 0 # Verdadero positivo, coincide
    FP = 0 # Falso Positivo, se detecto un objeto de mas que no esta presente en la verdad basica 
    FN = 0 # Falso negativo, no se detecto un objeto que debia ser detectado

    TP_acc = 0
    FP_acc = 0
    FN_acc = 0
    
    #print(d[0]['bbox'])
    for i in range(0, n):
        for j in range(0, 300):
            if d[i*300+j]['score'] > 0.50:
                detection_pr.append(d[i*300+j]['bbox'])
                cont += 1

        for j in anno:
            if j['image_id'] == i:
                detection_gt.append(j['bbox'])
        
        for j in range(0,len(detection_gt)):
            iou = 0
            if len(detection_pr) == 0:
                FN += 1
                continue
            
            for k in range(0,len(detection_pr)):
                aux = calcular_iou_coco(detection_pr[k], detection_gt[j])
                if aux > iou:
                    iou = aux

            if iou > 0.50:
                TP += 1
            else:
                FN += 1

        FP = len(detection_pr) - len(detection_gt) + FN

        TP_acc += TP
        FP_acc += FP
        FN_acc += FN

        TP = 0
        FP = 0
        FN = 0

        detection_gt = []
        detection_pr = []

    print(cont)
    
    print("Verdadero positivo ", TP_acc)
    print("Falso positivo ", FP_acc)
    print("Verdadero negativo ", TP_acc - FP_acc)
    print("Falso negativo ", FN_acc)

    precision = TP_acc/(TP_acc+FP_acc)
    recall = TP_acc/(TP_acc+FN_acc)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("PRECISION: ",precision)
    print("RECALL: ",recall)
    print("F1-SCORE: ",f1_score)
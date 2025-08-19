from fastapi import FastAPI
from fastapi import Response, status

from pydantic import BaseModel
from typing import List
import image_process
import Transf_px_mm
import cliente
import base64
from pathlib import Path
import ftp_status
import time
from ultralytics import SAM, FastSAM
from camara_arriba import app2
import vzense_proces
global dims_global
dims_global = []
# model_path = "ftp_ck\sam2_t.pt"
# model = SAM(model_path).to("cuda")

class DataPoint(BaseModel):
    id: int
    x: float
    y: float
    angle: float
    diff_angle: float
    confirmation: bool

class DataPointVz(BaseModel):
    id: int
    x: float
    y: float
    z:float
    angle: float
    diff_angle: float
    confirmation: bool


class DataCenter(BaseModel):
    x: float
    y: float
class ImageResponse(BaseModel):
    clase: str
    dimensiones: list
    filename: str
    image_b64: str

app = FastAPI()

@app.on_event("startup")
def load_model():
    model_path = "ftp_ck/FastSam-x.pt"
    #model_path = "ftp_ck/sam2.1_l.pt"
    
    #model = SAM(model_path).to("cuda")
    
    model = FastSAM(model_path)
    app.state.sam_model = model
    print("✅ SAM model loaded on startup")

#
@app.get("/")
def root():
    return {"status": "OK - Server ON"}


@app.get("/puntos_bolsas", response_model=List[DataPoint])
def get_puntos_bolsas():

    try:
        t_i = time.time()
        model = app.state.sam_model
        cliente.download_with_custom_key()
        # 1) Segmentación y cálculo de dimensiones (tu función existing)
        dims = image_process.main(model = model, cantidad_bolsas=5, umbral_mascaras=9)  # [(w,h,cx,cy,angle), ...]
        t_despues_sdm = time.time()
        conf = image_process.ImgConfirmation(dims, 5,0.3, tam_min=350000, tam_max=600000)

        
        global dims_global
        dims_global = dims

        # 2) Transformar pixeles a mm
        centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims)
        #    centers_mm == [(x1,y1), (x2,y2), ...]
        # 3) Calcular diferencias de ángulo
        dif_angles = Transf_px_mm.diferencia_angles(dims)
        #    dif_angles == [(a1,da1), (a2,da2), ...]
        
        # 4) Empaquetar todos los puntos en una lista
        points: List[DataPoint] = []
        #print("centers_mm",len(centers_mm))
        #print("dif_angles",len(dif_angles))
        t = 0
        
        for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
            #print("t",t)
            t = t+1
            points.append(DataPoint(
                id=idx,
                x=x,
                y=y,
                angle=angle,
                diff_angle=diff_angle,
                confirmation= conf
            ))
        #print("points ====== ",points)
        t_e = time.time()
        print("timeee ", t_e-t_i)
        return points
    
    except Exception as e:
        print("no da:  ", e)
@app.get("/puntos_bolsas_prod2h", response_model=List[DataPoint])
def get_puntos_bolsas_prod2h():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main(model = model, cantidad_bolsas=5, umbral_mascaras=9)  # [(w,h,cx,cy,angle), ...]
    t_despues_sdm = time.time()
    conf = image_process.ImgConfirmation(dims, 5,0.25, tam_min= 450000, tam_max=700000)

    
    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    print("dims:     ", dims)
    #centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 630, 410, 335)1

    centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims)
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]
    
    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    #print("centers_mm",len(centers_mm))
    #print("dif_angles",len(dif_angles))
    t = 0
    
    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        #print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    #print("points ====== ",points)
    t_e = time.time()
    print("timeee ", t_e-t_i)
    return points

@app.get("/puntos_bolsas_prod3h", response_model=List[DataPoint])
def get_puntos_bolsas_prod3h():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main_prod3h(model = model, cantidad_bolsas=6, umbral_mascaras=11)  # [(w,h,cx,cy,angle), ...]
    t_despues_sdm = time.time()
    conf = image_process.ImgConfirmation(dims, 6,0.3)

    
    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    #centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 650, 450, 335) ### CCAMBIAR ESCALA CUANDO SE TENGA DIMENSIONES DE LA BOLSA

    centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims) 
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]
    
    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    #print("centers_mm",len(centers_mm))
    #print("dif_angles",len(dif_angles))
    t = 0
    
    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        #print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    #print("points ====== ",points)
    t_e = time.time()
    print("timeee ", t_e-t_i)
    return points


@app.get("/centro_pallet", response_model=List[DataCenter])
def get_centro_pallet():
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    model = app.state.sam_model
    dims = image_process.main_centros_pallet(model=model, cantidad_bolsas = 5, rango = 700000)  # [(w,h,cx,cy,angle), ...]

    # 2) Transformar pixeles a mm
    centers_mm = Transf_px_mm.transform_pallet_center(dims,335, 1200, 1000)


    # 4) Empaquetar todos los puntos en una lista
    points: List[DataCenter] = []
    print("centers_mm",len(centers_mm))

    points.append(DataCenter(
            x=centers_mm[0],
            y=centers_mm[1]
        ))
    print("center ====== ",points)
    return points

@app.get("/centro_pallet_oregano", response_model=List[DataCenter])
def get_centro_pallet_oregano():
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    model = app.state.sam_model
    dims = image_process.main_centros_pallet(model=model, cantidad_bolsas = 5, rango=900000)  # [(w,h,cx,cy,angle), ...]

    # 2) Transformar pixeles a mm
    centers_mm = Transf_px_mm.transform_pallet_center(dims,335, 1200, 1000)


    # 4) Empaquetar todos los puntos en una lista
    points: List[DataCenter] = []
    print("centers_mm",len(centers_mm))

    points.append(DataCenter(
            x=centers_mm[0],
            y=centers_mm[1]
        ))
    print("center ====== ",points)
    return points


@app.get("/centro_pallet_6bolsas", response_model=List[DataCenter])
def get_centro_pallet_6bolsas():
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    model = app.state.sam_model
    dims = image_process.main_centros_pallet(model=model, cantidad_bolsas = 6, rango=700000)  # [(w,h,cx,cy,angle), ...]

    # 2) Transformar pixeles a mm
    centers_mm = Transf_px_mm.transform_pallet_center(dims,335, 1200, 1000)


    # 4) Empaquetar todos los puntos en una lista
    points: List[DataCenter] = []
    print("centers_mm",len(centers_mm))

    points.append(DataCenter(
            x=centers_mm[0],
            y=centers_mm[1]
        ))
    print("center ====== ",points)
    return points

@app.get("/entrenamiento", response_model=ImageResponse)
def get_entrenamiento():
    # 1) Ruta de tu imagen
    imagen_path = Path(r"D:\CIRAL\VISION\ftp_images\imgSend.bmp")

    # 2) Leemos y codificamos en base64
    with imagen_path.open("rb") as f:
        b64_bytes = base64.b64encode(f.read())
    img_b64 = b64_bytes.decode("utf-8")
    global dims_global
    # 3) Devolvemos JSON con nombre y contenido
    return ImageResponse(
        clase='harina prod 1',
        dimensiones=dims_global,
        filename=imagen_path.name,
        image_b64="img_b64"
    )

@app.get("/ftp_status")
def get_ftp_status(response: Response):
    activo = ftp_status.ftp_esta_activo("172.19.69.93", 2121)

    if activo:
        response.status_code = status.HTTP_200_OK
        return {"ftp_status": "ON"}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ftp_status": "OFF"}
    


@app.get("/puntos_bolsas_sal", response_model=List[DataPoint])
def get_puntos_bolsas_sal():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    #dims = image_process.main(model = model, cantidad_bolsas=6, umbral_mascaras=11)  # [(w,h,cx,cy,angle), ...]

    dims = image_process.main_bolsa(model = model, cantidad_bolsas=6, umbral_mascaras=11)  # [(w,h,cx,cy,angle), ...]
    conf = image_process.ImgConfirmation(dims, 6,0.2 , tam_max=600000 , tam_min= 400000)
    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims)
    #centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 600, 400, 335)
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]
    
    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    #print("centers_mm",len(centers_mm))
    #print("dif_angles",len(dif_angles))
    t = 0
    
    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        #print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    #print("points ====== ",points)
    t_e = time.time()
    print("timeee ", t_e-t_i)
    return points


@app.get("/puntos_bolsas_oregano", response_model=List[DataPoint])
def get_puntos_bolsas_oregano():
    import time
    t_i = time.time()
    max_retries = 10

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Intento {attempt}...")
            model = app.state.sam_model
            if model is None:
                print("El modelo SAM no está cargado.")
                continue

            cliente.download_with_custom_key()

            dims = image_process.main(model=model, cantidad_bolsas=5, umbral_mascaras=9)
            if dims is None:
                print("dims es None.")
                continue

            conf = image_process.ImgConfirmation(dims, 5, 0.2)
            global dims_global
            dims_global = dims

            centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims)
            dif_angles = Transf_px_mm.diferencia_angles(dims)

            points: List[DataPoint] = []
            for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles)):
                points.append(DataPoint(
                    id=idx,
                    x=x,
                    y=y,
                    angle=angle,
                    diff_angle=diff_angle,
                    confirmation=conf
                ))

            t_e = time.time()
            print("Duración total:", t_e - t_i)
            return points

        except Exception as e:
            print(f"Error en intento {attempt}: {e}")
            time.sleep(0.5)

    # Si fallaron todos los intentos
    print("Todos los intentos fallaron. Retornando lista vacía.")
    return []



@app.get("/puntos_bolsas_prod2s", response_model=List[DataPoint])
def get_puntos_bolsas_prod2s():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main_prod3h(model = model, cantidad_bolsas=5, umbral_mascaras=9)  # [(w,h,cx,cy,angle), ...]
    t_despues_sdm = time.time()
    conf = image_process.ImgConfirmation(dims, 5,0.3)

    
    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    #centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 650, 450, 335)
    centers_mm = Transf_px_mm.convertir_centros_px_a_mm(dims)
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]
    
    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    #print("centers_mm",len(centers_mm))
    #print("dif_angles",len(dif_angles))
    t = 0
    
    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        #print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    #print("points ====== ",points)
    t_e = time.time()
    print("timeee ", t_e-t_i)
    return points


@app.get("/bolsa", response_model=List[DataPoint])
def get_bolsa():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main_bolsa(model = model, cantidad_bolsas=1, umbral_mascaras=2)  # [(w,h,cx,cy,angle), ...]
    t_despues_sdm = time.time()
    conf = image_process.ImgConfirmation(dims, 5,0.3)

    
    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 650, 450, 335)
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]
    
    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    #print("centers_mm",len(centers_mm))
    #print("dif_angles",len(dif_angles))
    t = 0
    
    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        #print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    #print("points ====== ",points)
    t_e = time.time()
    print("timeee ", t_e-t_i)
    return points





@app.get("/all_mask")
def get_all_mask():
    t_i = time.time()
    model = app.state.sam_model
    cliente.download_with_custom_key()
    image_process.mostrar_todas_mascaras(model)
    return {"status": "OK"}


@app.get("/vzense_pallet")
def get_vzense_pallet():
    return app2.detectar_objetos()




######################## VZSENSE SEGMENTACION #########################


class Posicion(BaseModel):
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float

# Variable global para guardar la última posición
posicion_actual: Posicion | None = None

# POST para guardar la posición del robot
@app.post("/posicion")
def guardar_posicion(pos: Posicion):
    global posicion_actual
    posicion_actual = pos
    print("Posicion del robot : " , [posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a])
    return {"status": "posicion recibida", "posicion": pos}



@app.get("/puntos_bolsas_vzsense_harina3", response_model=List[DataPointVz])
def get_puntos_bols_vzsense_sal1():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a, min_area=90000, max_area = 150000, altura_foto = 1650)
    conf = image_process.ImgConfirmation(dims,6,0.7 ,tam_min= 90000, tam_max=180000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points



@app.get("/puntos_bolsas_vzsense", response_model=List[DataPointVz])
def get_puntos_bols_vzsense():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a,  min_area = 80000, max_area = 180000,altura_foto = 1650)
    conf = image_process.ImgConfirmation(dims, 5,0.3,tam_min = 80000, tam_max = 190000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points

@app.get("/puntos_bolsas_vzsense2h", response_model=List[DataPointVz])
def get_puntos_bols_vzsense():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a, altura_foto= 1630)
    conf = image_process.ImgConfirmation(dims, 5,0.3, tam_min= 90000, tam_max=250000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points

@app.get("/puntos_bolsas_vzsense_sal2", response_model=List[DataPointVz])
def get_puntos_bols_vzsense_sal2():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a, min_area=90000, max_area = 190000)
    conf = image_process.ImgConfirmation(dims,6,0.5 ,tam_min= 90000, tam_max=150000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points


@app.get("/puntos_bolsas_vzsense_sal4", response_model=List[DataPointVz])
def get_puntos_bols_vzsense_sal4():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a, min_area=90000, max_area = 400000)
    conf = image_process.ImgConfirmation(dims,5,0.5 ,tam_min= 90000, tam_max=400000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points

@app.get("/puntos_bolsas_vzsense_sal1", response_model=List[DataPointVz])
def get_puntos_bols_vzsense_sal1():
    #print("===================", posicion_actual)
    centers_mm, dif_angles, dims = vzense_proces.geCenters(posicion_actual.x, posicion_actual.y, posicion_actual.z, posicion_actual.c, posicion_actual.b, posicion_actual.a, min_area=90000, max_area = 150000)
    conf = image_process.ImgConfirmation(dims,5,0.7 ,tam_min= 90000, tam_max=180000)
    
    print("==========angles========", dif_angles)
    print("==========centers========", centers_mm)
    points: List[DataPointVz] = []
    for idx, ((x, y,z), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        points.append(DataPointVz(
            id=idx,
            x=x,
            y=y,
            z=z,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    return points
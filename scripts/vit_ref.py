import glob,os,numpy as np,cv2,onnxruntime as ort,json
R='/workspace/mp/'; D=R+'vit/test'; os.makedirs(D,exist_ok=True)
hg=sorted(glob.glob(R+'images/*.jp*g')+glob.glob(R+'images/*.png'))   # any photos you like
s=ort.InferenceSession(R+'vit/deit_tiny.onnx',providers=['CPUExecutionProvider'])
mean=np.array([123.675,116.28,103.53],np.float32); std=np.array([58.395,57.12,57.375],np.float32)
lab=json.load(open(R+'vit/config.json'))['id2label']
for k,f in enumerate(hg):
    im=cv2.imread(f); h,w=im.shape[:2]; sc=224/min(h,w); im=cv2.resize(im,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA); h,w=im.shape[:2]; y,x=(h-224)//2,(w-224)//2; im=im[y:y+224,x:x+224]
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); cv2.imwrite(f'{D}/v_{k:02d}.png',im)
    rgb.tofile(f'{D}/v_{k:02d}.in.bin')   # raw uint8 HWC RGB (preproc on NPU)
    x=((rgb.astype(np.float32)-mean)/std).transpose(2,0,1)[None]
    lg=s.run(None,{'pixel_values':x})[0].ravel(); np.save(f'{D}/v_{k:02d}.ref.npy',lg)
    print(k,[lab[str(i)].split(',')[0] for i in np.argsort(-lg)[:3]])

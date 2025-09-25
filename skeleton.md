root/
├─ .gitignore
├─ .fdiivenv
├─ README.md
├─ requirements.txt
├─ modules/
│  ├─ preprocess/
│  │  └─ preprocess.py        
│  ├─ face_detection/
│  │  └─ detect_and_mask.py   
│  ├─ synthetic_face_generation/
│  │  ├─ generate_dtwin_brushnet.py   
│  │  ├─ generate_dtwin_cloud.py      
│  ├─ video_integration/
│  │  └─ animate_fomm.py      
│  ├─ evaluation/
│  │  └─ evaluate.py          
│  └─ deployment/
│     └─ deploy_local.sh      
├─ tools/
│   └─ utils.py                
└─ jobs/
   └─ .gitkeep     
   
# create new env (.fdiivenv)
# python -m venv .fdiivenv
# source .fdiivenv/bin/activate
# pip install -r requirements.txt                 

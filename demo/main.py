# 导入所需库
import websocket  # NOTE: 需要安装websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io

# 设置服务器地址和客户端ID
server_address = "172.168.16.66:8188"
client_id = str(uuid.uuid4())


# 定义向服务器发送提示的函数
def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


# 定义从服务器下载图像数据的函数
def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


# 定义获取历史记录的函数
def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())


# 定义通过WebSocket接收消息并下载图像的函数
def get_images(ws, prompt):
    print("xxxxxxxxxxxxxxx")
    prompt1 = queue_prompt(prompt)
    print(prompt1)

    prompt_id = prompt1['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # 执行完成
        else:
            continue  # 预览是二进制数据

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


# 示例JSON字符串，表示要使用的提示
prompt_text = """
{
  "8": {
    "inputs": {
      "samples": [
        "329",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "14": {
    "inputs": {
      "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint Base"
    }
  },
  "222": {
    "inputs": {
      "pixels": [
        "392",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "329": {
    "inputs": {
      "seed": 742881690607952,
      "steps": 4,
      "cfg": 3,
      "sampler_name": "dpmpp_sde",
      "scheduler": "karras",
      "denoise": 0.55,
      "model": [
        "393",
        0
      ],
      "positive": [
        "336",
        1
      ],
      "negative": [
        "336",
        2
      ],
      "latent_image": [
        "222",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "336": {
    "inputs": {
      "weight": 1.2,
      "start_at": 0,
      "end_at": 0.9500000000000001,
      "instantid": [
        "337",
        0
      ],
      "insightface": [
        "338",
        0
      ],
      "control_net": [
        "339",
        0
      ],
      "image": [
        "340",
        0
      ],
      "model": [
        "14",
        0
      ],
      "positive": [
        "368",
        0
      ],
      "negative": [
        "369",
        0
      ],
      "image_kps": [
        "392",
        0
      ]
    },
    "class_type": "ApplyInstantID",
    "_meta": {
      "title": "Apply InstantID"
    }
  },
  "337": {
    "inputs": {
      "instantid_file": "ip-adapter.bin"
    },
    "class_type": "InstantIDModelLoader",
    "_meta": {
      "title": "Load InstantID Model"
    }
  },
  "338": {
    "inputs": {
      "provider": "CPU"
    },
    "class_type": "InstantIDFaceAnalysis",
    "_meta": {
      "title": "InstantID Face Analysis"
    }
  },
  "339": {
    "inputs": {
      "control_net_name": "diffusion_pytorch_model.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "340": {
    "inputs": {
      "image": "002IWC3ygy1hhcdx2z1eqj60u0190n2h02 (1).jpg",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "face"
    }
  },
  "345": {
    "inputs": {
      "image": "007rSCv8gy1hn00iwonnyj30u019278u.jpg",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "pose"
    }
  },
  "368": {
    "inputs": {
      "text": "",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "369": {
    "inputs": {
      "text": "",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "377": {
    "inputs": {
      "crop_blending": 0.25,
      "crop_sharpening": 0,
      "image": [
        "345",
        0
      ],
      "crop_image": [
        "8",
        0
      ],
      "crop_data": [
        "383",
        1
      ]
    },
    "class_type": "Image Paste Face",
    "_meta": {
      "title": "Image Paste Face"
    }
  },
  "378": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "377",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "380": {
    "inputs": {
      "categories": "face,",
      "confidence_threshold": 0.1,
      "iou_threshold": 0.1,
      "box_thickness": 2,
      "text_thickness": 2,
      "text_scale": 1,
      "with_confidence": true,
      "with_class_agnostic_nms": false,
      "with_segmentation": true,
      "mask_combined": true,
      "mask_extracted": true,
      "mask_extracted_index": 0,
      "yolo_world_model": [
        "381",
        0
      ],
      "esam_model": [
        "382",
        0
      ],
      "image": [
        "345",
        0
      ]
    },
    "class_type": "Yoloworld_ESAM_Zho",
    "_meta": {
      "title": "🔎Yoloworld ESAM"
    }
  },
  "381": {
    "inputs": {
      "yolo_world_model": "yolo_world/l"
    },
    "class_type": "Yoloworld_ModelLoader_Zho",
    "_meta": {
      "title": "🔎Yoloworld Model Loader"
    }
  },
  "382": {
    "inputs": {
      "device": "CUDA"
    },
    "class_type": "ESAM_ModelLoader_Zho",
    "_meta": {
      "title": "🔎ESAM Model Loader"
    }
  },
  "383": {
    "inputs": {
      "padding": 15,
      "region_type": "minority",
      "mask": [
        "380",
        1
      ]
    },
    "class_type": "Mask Crop Region",
    "_meta": {
      "title": "Mask Crop Region"
    }
  },
  "384": {
    "inputs": {
      "width": [
        "383",
        6
      ],
      "height": [
        "383",
        7
      ],
      "x": [
        "383",
        3
      ],
      "y": [
        "383",
        2
      ],
      "image": [
        "345",
        0
      ]
    },
    "class_type": "ImageCrop",
    "_meta": {
      "title": "ImageCrop"
    }
  },
  "387": {
    "inputs": {
      "images": [
        "392",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "388": {
    "inputs": {
      "images": [
        "380",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "391": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "392": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1,
      "image": [
        "384",
        0
      ]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {
      "title": "ImageScaleToTotalPixels"
    }
  },
  "393": {
    "inputs": {
      "hard_mode": true,
      "boost": true,
      "model": [
        "336",
        0
      ]
    },
    "class_type": "Automatic CFG",
    "_meta": {
      "title": "Automatic CFG"
    }
  }
}
"""

if __name__ == '__main__':
    # 将示例JSON字符串解析为Python字典，并根据需要修改其中的文本提示和种子值
    prompt = json.loads(prompt_text)
    prompt["345"]["inputs"]["image"] = "./WechatIMG108.jpg"
    prompt["340"]["inputs"]["image"] = "./WechatIMG109.jpg"


    # 创建一个WebSocket连接到服务器
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

    # 调用get_images()函数来获取图像
    images = get_images(ws, prompt)

    # 显示输出图像（这部分已注释掉）
    # Commented out code to display the output images:
    #
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            image.save("local_image.jpg")

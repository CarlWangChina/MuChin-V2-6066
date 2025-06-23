import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sunoapi.service.dewatermark
import sunoapi.service.suno_api_call
import multiprocessing
import sunoapi.config_loader as config

if __name__ == "__main__":
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    upload_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()

    suno = sunoapi.service.suno_api_call.TaskManager(task_queue, result_queue, error_queue, 10)
    suno.start()

    dewm = sunoapi.service.dewatermark.TaskManager(result_queue, upload_queue, error_queue, device_list=[0, 1, 2, 3])
    dewm.start()

    suno.add_task({
        "title": "越人语天姥，云霞明灭或可睹",
        "style": "instrument, piano, harp, male vocal",
        "lyric": "海客谈瀛洲，烟涛微茫信难求；\n越人语天姥，云霞明灭或可睹。\n天姥连天向天横，势拔五岳掩赤城。\n天台四万八千丈，对此欲倒东南倾。\n我欲因之梦吴越，一夜飞度镜湖月。\n湖月照我影，送我至剡溪。\n谢公宿处今尚在，渌水荡漾清猿啼。\n脚著谢公屐，身登青云梯。\n半壁见海日，空中闻天鸡。\n千岩万转路不定，迷花倚石忽已暝。\n熊咆龙吟殷岩泉，栗深林兮惊层巅。\n云青青兮欲雨，水澹澹兮生烟。\n列缺霹雳，丘峦崩摧。\n洞天石扉，訇然中开。\n青冥浩荡不见底，日月照耀金银台。\n霓为衣兮风为马，云之君兮纷纷而来下。\n虎鼓瑟兮鸾回车，仙之人兮列如麻。\n忽魂悸以魄动，恍惊起而长嗟。\n惟觉时之枕席，失向来之烟霞。\n世间行乐亦如此，古来万事东流水。\n别君去兮何时还？\n且放白鹿青崖间，须行即骑访名山.",
        "mid": "100000000"
    })

    while True:
        print(upload_queue.get())
        break
import json
from mapy import db
from Code_for_MuChin_AP.backend_api.mapy.util.other import from_session_map

def output_data():
    result = []

    sql = f'''SELECT result.mid , lrc_song , lrc_section , lrc_rhyme, gpt_sum_content FROM result LEFT JOIN work_4_qa ON result.mid = work_4_qa.mid WHERE type = 1 AND lrc_extra_lan = '' AND special = 0'''
    res_list = db.session.execute(db.text(sql)).all()
    for iid, res in enumerate(res_list):
        print(str(iid) + "/" + str(len(res_list)))
        song_list = res.lrc_song.split('\n')
        mid = res.mid
        data1 = ""
        data2 = ""
        data3 = res.gpt_sum_content
        new_flag_section = res.lrc_section.split(',')
        print(res.mid)
        for new_section in new_flag_section:
            print(new_section)
            sub_sec = new_section[0:2]

            data1 += f'({from_session_map(sub_sec[0])} {str(ord(sub_sec[1]) - 96)}\n'
            data2 += f'({from_session_map(sub_sec[0])} {str(ord(sub_sec[1]) - 96)}\n'
            for index, item in enumerate(song_list):
                new_sub_sec = new_section[2 * index: 2 * index + 2]
                if sub_sec != new_sub_sec:
                    sub_sec = new_sub_sec
                    data1 += f'({from_session_map(sub_sec[0])} {str(ord(sub_sec[1]) - 96)}\n'
                    data2 += f'({from_session_map(sub_sec[0])} {str(ord(sub_sec[1]) - 96)}\n'
                rhy = ''.join(['c' if char != ' ' else char for char in item])
                if res.lrc_rhyme[index] != 'Z':
                    rhy = rhy[0:-1] + 'R'
                data1 += f'{item}\n'
                data2 += f'{rhy}\n'
            result.append({
                'mid': mid,
                'data1': data1.strip(),
                'data2': data2.strip(),
                'data3': data3.strip(),
            })
    with open("mapy/music/re.json", 'w') as ff:
        json.dump(result, ff, ensure_ascii=False, indent=4)
inputOcr = '경기도 안양시 동안구 시민대로248번길 25, 505호, 506호'
# 기존 code

def address_parser_kor(kor_address):
    address_dict = {}
    kor_address = kor_address.strip()
    import re

    pattern = re.compile(r'\(.*\)?')
    searching = re.search(pattern, kor_address)
    if searching is not None:
        searching = searching.group()
        main = kor_address.replace(searching, "")
    else:
        main = kor_address

    main = main.split(",")
    if len(main) == 2:
        etc = main[1]
    main = main[0].strip()
    main = main.replace(" ", "")

    p = re.compile(r'.{1,6}(도|별시|역시|치시)')
    sido_match = re.match(p, main)
    if sido_match is not None:
        sido = sido_match.group()
    else:
        sido = ""
    main = main.replace(sido, "")

    if main[2] == "구" or main[2] == "군" or main[2] == "시":
        gugunsi = main[0:3]
    elif main[3] == "구" or main[3] == "군" or main[3] == "시":
        gugunsi = main[0:4]
    elif main[4] == "구" or main[4] == "군" or main[4] == "시":
        gugunsi = main[0:5]
    else:
        gugunsi = ""
    main = main.replace(gugunsi, "")

    p = re.compile(r'(?<=로|길)\d+-*\d*$')  # 뒷숫자 찾기 완성
    road_num = re.search(p, main)
    if road_num is not None:
        road_num = road_num.group()
    else:
        road_num = ""
    main = main.replace(road_num, "")

    road = ""
    if gugunsi[-1] == "구" and ((main[-1] == "길") or (main[-1] == "로")):
        road = main
        main = main.replace(road, "")
    else:
        p = re.compile(r'.{1,4}(구|읍|면)')
        if re.match(p, main) is not None:
            gugunsi_extra = re.match(p, main).group()
            gugunsi = gugunsi + " " + gugunsi_extra
            main = main.replace(gugunsi_extra, "")
            road = main
            main = main.replace(road, "")
        else:
            road = main
            main = main.replace(road, "")

    address_dict['sido'] = sido
    address_dict['gugunsi'] = gugunsi
    address_dict['road'] = road + " " + road_num
    # address_dict['etc'] = etc.strip() if "etc" in locals() else ""
    address_dict['additional'] = searching if searching is not None else ""

    # if address_dict['etc'] == "" and address_dict['additional'] =="":
    result = f"{address_dict['sido']}${address_dict['gugunsi']}${address_dict['road']}{address_dict['additional']}"
    # else:
    #     result = f"{address_dict['sido']}${address_dict['gugunsi']}${address_dict['road']}{address_dict['additional']}"

    return result

field_value=address_parser_kor(inputOcr)

print(field_value)
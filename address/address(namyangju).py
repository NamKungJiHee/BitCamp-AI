inputOcr = '인천광역시 부평구 안남로402번길 76(청천동, (주)이피아이티)'
# 경기도 남양주
# 시도/구군시/도로명건물명

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
    if  gugunsi[-1] == "구" and ((main[-1] == "길") or (main[-1] == "로")):
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
    address_dict['additional'] = searching if searching is not None else ""

    #print(sido,'======' ,gugunsi)
    
    if sido[2] == "도" and gugunsi[3] == "시":
        gugunsi = gugunsi.split('시')
        #print(gugunsi[0]) # 남양주
       # print(gugunsi[1]) #  진건읍
        
        gugunsi2 = gugunsi[1]
        gugunsi = sido + gugunsi[0]
       # print(gugunsi2)
       # print(gugunsi) # 경기도남양주
       # gugunsi = address_dict['sido'] + address_dict['gugunsi']
        result =  f"{gugunsi}${gugunsi2}${address_dict['road']}{address_dict['additional']}"
        
        #print(gugunsi)
        #print(address_dict['sido']) #경기도
        #print(address_dict['gugunsi']) #남양주시 진건읍
        #print(results) #경기도남양주시 진건읍
        #print(result) #경기도남양주시 진건읍$진건오남로390본갈169-47
        
    else:
    # if address_dict['etc'] == "" and address_dict['additional'] =="":
        result = f"{address_dict['sido']}${address_dict['gugunsi']}${address_dict['road']}{address_dict['additional']}"

    return result

field_value=address_parser_kor(inputOcr)

print(field_value)
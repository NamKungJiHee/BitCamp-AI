inputOcr = 'Seoultikbulsi Geumcheongu gasandigital2ro 179(gasandong, Lottlecenter)'
'''
Seoultikbulsi Yongsangu Hangangdaero 92(Hangangro2ga, LS Yongsantower)
Daejeonsi Daedeokgu Galjeondong
Gyeonggido Namyangjusi Jingeoneup JingeonOnamro390beongil 169-47
Incheongwangyeoksi Bupyeonggu Annamro402beongil 76(Cheongcheondong, (ju)it)
Incheongwangyeoksi seogu jungbongdaero 198beongil 2(gajadong)
Seoultikbulsi jongrogu Yulgokro2gil 25, 14cheung(susongdong)
Seoultikbulsi mapogu worldcupro 96, 4cheung(Seogyodong, younghunbuilding)
Seoultikbulsi Seochogu gangnamdaero 465, 18 cheung(seochodong, gyobotower)
Daejeongwangyeoksi Yuseonggu bansukro112beongil 1, 2cheung(bansukdong)
Seoultikbulsi Nowongu gwangunro 20(Wolgyedong)
Seoultikbulsi junggu donghoro 387-2, 1,3,5cheung(bangsandong)
Seoultikbulsi songpagu beopwonro 114, Adong 13cheung 1305ho(munjungdong, amstate)
Seoultikbulsi jongrogu jongro 1(jongro1ga, gyobobuilding 20cheung)
Seoultikbulsi Seochogu tabongro 114, 8cheung 810ho yangjaeheobu(wumyeondong, hankuckgyeowondanchachongyeonhaphoe)
Seoultikbulsi Mapogu worldcupbukro 396(sangamdong, nurisquarebijinisutawar 14cheung)
Seoultikbulsi gangnamgu samsungro 524, 601ho(samsungdong, sehwabuilding)
Seoultikbulsi jongrogu jongro33gil 15(yeonjidong, yeongangbuilding 8 cheung)
Incheongwangyeoksi Michuholgu inharo 100(yonghyeondong, inhadahakgyo)
Seoultikbulsi jongrogu Yulgokro2gil 25, 14 cheung(susongdong)
Seoultikbulsi Gangnamgu Teheranno 606(dachidong)
Seoultikbulsi Geumcheongu butkkotro 298, 1005ho(gasandong, darungposototower6cha)
Seoultikbulsi Geumcheongu gasandigital2ro 179(gasandong, Lottlecenter)
'''
def address_parser_eng(eng_address):
    address_dict = {}
    main = eng_address.split("!")

    #print(main) # ['Gyeonggi-do Gwangjusi jungu nanbangdong']
  
    address_dict['sido'] = main[0].split('si')
    #print(address_dict['sido']) # ['Gyeonggi-do Gwangju', ' jungu nanbangdong']
    address_dict['gugunsi'] = address_dict['sido'][1]
    address_dict['sido'] = address_dict['sido'][0]
    #print(address_dict['sido']) # Gyeonggi-do Gwangju
    address_dict['sido'] = address_dict['sido'] + 'si'
   #print(address_dict['sido']) # Gyeonggido Namyangjusi
   # print(address_dict['gugunsi']) # Jingeoneup JingeonOnamro390beongil 169-47
    
    if 'gu' in address_dict['gugunsi']:
        address_dict['gugunsi'] = address_dict['gugunsi'].split('gu')
    else: 
        address_dict['gugunsi'] = address_dict['gugunsi'].split('eup')
    
    #print(address_dict['gugunsi']) # [' Jingeon', ' JingeonOnamro390beongil 169-47']
      
    address_dict['additional'] = address_dict['gugunsi'][1]
   # print(address_dict['gugunsi'][1])
    address_dict['gugunsi'] = address_dict['gugunsi'][0]
   # print(address_dict['gugunsi']) #   Yongsan

   # print(address_dict['additional']) # nanbangdong
   # print(main[0])
    if 'gu' in main[0]:
        address_dict['gugunsi'] = address_dict['gugunsi'] + 'gu'
        
    else:
        address_dict['gugunsi'] = address_dict['gugunsi'] + 'eup'
    
    
  #  print(address_dict['gugunsi']) 
    
    #print(address_dict['additional'])
    result = f"{address_dict['sido']}${address_dict['gugunsi']}${address_dict['additional']}"
    return result

field_value = address_parser_eng(inputOcr) 
print(field_value) 
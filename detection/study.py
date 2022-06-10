from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import time
import urllib.request
import os
import shutil
import dlib,cv2                             
import numpy as np

while True:
    
    errorcode = []       
    
    
    while True:
        keyword = input('검색할 키워드를 입력하세요(exit -> 종료) : ')
        if keyword == 'exit':
            exit() 
        saveword = input('저장할 폴더명을 입력하세요(영어로) : ')
                  
        try:
            os.makedirs(f'humans/{saveword}/Original',exist_ok=False)
            os.makedirs(f'humans/{saveword}/Download',exist_ok=False)
            os.makedirs(f'humans/{saveword}/real',exist_ok=False)
            os.makedirs(f'humans/{saveword}/unreal',exist_ok=False) 
            break
        except:
            errorcode.append(f'{saveword}생성 오류')
            d = int(input('이미 있는 키워드입니다 삭제 후 재실행 할까요? 1 - 네 / 2 - 아니오 : '))
            if d == 1:
                shutil.rmtree(f'humans/{saveword}')
                continue
            elif d == 2:
                continue       
            
    n = int(input('저장할 사진 장수를 입력해주세요 : '))
    
    
    ne = round(n/33)                                        
                                                                                                                                         
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome('C://chromedriver.exe', options=options)                                 
    
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")                                            
    elem = driver.find_element_by_name("q")                                                             
    elem.send_keys(f'{keyword}')                                                                        
    elem.send_keys(Keys.RETURN)                                                                        
    images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")                                       
    if len(images) < n+ne:                                                                              
        while True:                                                                                    
            last_height = driver.execute_script("return document.body.scrollHeight") # 스크롤 높이 가져옴                  
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # 끝까지 스크롤 다운                   
            time.sleep(1)          # document.body.scrollHeight: 페이지 끝까지 가려면                                                                          
            new_height = driver.execute_script("return document.body.scrollHeight") # 스크롤 다운 후 스크롤 높이 다시 가져옴                   
            images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")                               
            print(f'{keyword}사진 현재{len(images)}장 로드, {n+ne}장까지 로드하기 위해 스크롤 중................') 
            
            
        
            if len(images) >= n+ne:                                                                     
                print(f'.........................................현재{len(images)}장까지 로드하여 이제 저장을 시작합니다')
                break                                                                                   
            else:                                                                                      
                if new_height == last_height:                                                           
                    try:                                                                                        
                        driver.find_element_by_css_selector('.mye4qd').click()                          
                    except:                                                                                     
                        errorcode.append('스크롤 오류. 더 로드할 사진이 없습니다.')       
                        print('스크롤 오류. 더 로드할 사진이 없습니다.')                
                        break                                                                           
                last_height = new_height                                                               
    time.sleep(3)     
    count = 1
    totalstart = time.time()
    for image in images:
        try:                      
            image.click()                       
            start = time.time()
            imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img').get_attribute('src')                
            print(f'{count} / {n}, {keyword}사진 다운로드 중......... Download time : '+str(time.time() - start)[:5]+' 초')                 
            count = str(count).zfill(len(str(n+ne)))                        
            urllib.request.urlretrieve(imgUrl, f"humans/{saveword}/Download/{count}.jpg")    
            time.sleep(0.5)                                                     
            count = int(count.lstrip('0'))  
            if count == n :             
                break
            count += 1                   
        except:                    
            errorcode.append(f'{count}번째 저장오류 다음사진을 저장합니다.')      
            print(f'{count}번째 저장오류 다음사진을 저장합니다.')   
    totalend = str(time.time() - totalstart)[:5]   
    if count < n:          
        errorcode.append(f'더 이상의 사진이 없기때문에 {n}장까지 다운로드하지 못하고 {count-1}장까지만 저장했습니다.')      
        
    print(f'------------------다운로드시간 {totalend}초-----------------------')
    driver.close()
    print(f'-------------{saveword} 폴더생성이 완료되었습니다.-----------------')
    print('------------------원본으로 사용할 사진을 몇장정도 복사해서 Original 폴더에 넣어주세요 ----------------')
    os.startfile(f'humans\{saveword}\Download')

    input('엔터 누르면 다음 작업을 시작합니다.')

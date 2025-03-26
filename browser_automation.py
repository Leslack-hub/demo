# -*- coding: utf-8 -*-
# 导入所需的库
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
import time
import sys
# import os
import random
from fake_useragent import UserAgent
import undetected_chromedriver as uc

# 设置Chrome浏览器
print("正在启动浏览器...")

# 使用undetected_chromedriver来绕过反爬虫检测
try:
    # 方法1：使用undetected_chromedriver
    print("尝试使用undetected_chromedriver...")
    options = uc.ChromeOptions()

    # 移除无头模式，但设置窗口位置到屏幕外或次要显示器
    # options.add_argument("--headless")  # 禁用无头模式
    options.add_argument("--window-position=2000,0")  # 将窗口移到屏幕外或次要显示器
    options.add_argument("--window-size=1280,720")  # 设置适当的窗口大小

    # 添加随机用户代理
    try:
        ua = UserAgent()
        user_agent = ua.random
    except:
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    options.add_argument(f'--user-agent={user_agent}')

    # 添加其他选项来模拟真实浏览器
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")  # 禁用扩展
    options.add_argument("--no-sandbox")  # 禁用沙盒模式

    # 添加一些额外的参数来绕过检测
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")

    # 使用undetected_chromedriver创建驱动
    driver = uc.Chrome(options=options)

except Exception as e:
    print(f"undetected_chromedriver失败: {e}")
    print("尝试使用标准Selenium方法...")

    # 方法2：使用标准Selenium但添加反检测措施
    options = webdriver.ChromeOptions()

    # 移除无头模式，但设置窗口位置到屏幕外或次要显示器
    # options.add_argument("--headless")  # 禁用无头模式
    options.add_argument("--window-position=2000,0")  # 将窗口移到屏幕外或次要显示器
    options.add_argument("--window-size=1280,720")  # 设置适当的窗口大小

    # 添加随机用户代理
    try:
        ua = UserAgent()
        user_agent = ua.random
    except:
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    options.add_argument(f'--user-agent={user_agent}')

    # 添加反爬虫检测选项
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # 添加其他选项
    options.add_argument("--disable-extensions")  # 禁用扩展
    options.add_argument("--disable-gpu")  # 禁用GPU加速
    options.add_argument("--no-sandbox")  # 禁用沙盒模式
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")

    # 创建驱动
    driver = webdriver.Chrome(options=options)

    # 执行JavaScript来修改webdriver属性，进一步绕过检测
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_script("window.navigator.chrome = {runtime: {}}")
    driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']})")
    driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")


def random_sleep(min_seconds=1, max_seconds=5):
    """随机等待一段时间，模拟人类行为"""
    time.sleep(random.uniform(min_seconds, max_seconds))


def main(message):
    succeed = False
    try:
        # 设置页面加载超时
        driver.set_page_load_timeout(30)

        # 打开网站前先访问一个常见网站，降低可疑性
        # print("先访问常见网站...")
        # driver.get("https://www.google.com")
        # random_sleep(2, 5)

        # 打开目标网站
        print("正在访问目标网站...")
        driver.get("https://lmarena.ai/")

        # 随机等待页面加载，模拟人类行为
        # random_sleep(10, 20)

        # 处理可能出现的服务条款同意框
        # print("检查是否存在服务条款同意框...")
        # 先在主文档中查找
        consent_button_found = False
        start_time = time.time()
        try:
            while time.time() - start_time < 30:
                # 方法1：尝试处理JavaScript alert弹窗
                try:
                    # 增加等待时间，确保有足够时间检测到弹窗
                    WebDriverWait(driver, 5).until(EC.alert_is_present())
                    alert = driver.switch_to.alert
                    print(f"检测到JavaScript弹窗: {alert.text}")
                    # 保存弹窗文本以便记录
                    alert_text = alert.text
                    # 接受弹窗
                    alert.accept()
                    print("已接受服务条款(JavaScript弹窗)")
                    # print(f"弹窗内容: {alert_text}")
                    # random_sleep(1, 2)  # 增加等待时间，确保弹窗完全关闭
                    consent_button_found = True
                    break
                except Exception as e:
                    print(f"没有检测到JavaScript弹窗或处理失败: {e}")

            if consent_button_found is False:
                print(f"处理失败")
                return

            # if consent_button_found is False:
            #     # 方法2：尝试处理HTML弹窗 - 查找常见的同意按钮
            #     consent_button_selectors = [
            #         # 通用同意按钮选择器
            #         (By.XPATH,
            #          "//button[contains(text(), 'Accept') or contains(text(), '同意') or contains(text(), 'Agree') or contains(text(), 'I agree') or contains(text(), 'OK') or contains(text(), '确定')]"),
            #         (By.XPATH,
            #          "//a[contains(text(), 'Accept') or contains(text(), '同意') or contains(text(), 'Agree') or contains(text(), 'I agree')]"),
            #         (By.XPATH,
            #          "//div[contains(text(), 'Accept') or contains(text(), '同意') or contains(text(), 'Agree') or contains(text(), 'I agree')]"),
            #         (By.XPATH,
            #          "//span[contains(text(), 'Accept') or contains(text(), '同意') or contains(text(), 'Agree') or contains(text(), 'I agree')]"),
            #         # 针对服务条款的特定按钮
            #         (By.XPATH,
            #          "//button[contains(@id, 'accept') or contains(@id, 'agree') or contains(@id, 'consent')]"),
            #         (By.XPATH,
            #          "//button[contains(@class, 'accept') or contains(@class, 'agree') or contains(@class, 'consent')]"),
            #         # 针对cookie同意按钮
            #         (By.XPATH, "//button[contains(text(), 'cookie') or contains(text(), 'Cookie')]"),
            #         # 针对服务条款特定文本的按钮
            #         (By.XPATH, "//button[contains(text(), 'terms') or contains(text(), 'Terms')]"),
            #         (By.XPATH, "//button[contains(text(), 'service') or contains(text(), 'Service')]"),
            #     ]
            #
            #     for selector_type, selector in consent_button_selectors:
            #         try:
            #             consent_button = WebDriverWait(driver, 2).until(
            #                 EC.element_to_be_clickable((selector_type, selector))
            #             )
            #             if consent_button:
            #                 print(f"找到同意按钮: {consent_button.text}")
            #                 # 使用JavaScript点击，有时比直接点击更可靠
            #                 driver.execute_script("arguments[0].click();", consent_button)
            #                 print("已点击同意按钮(HTML弹窗)")
            #                 random_sleep(2, 5)  # 等待弹窗关闭
            #                 consent_button_found = True
            #                 break
            #         except Exception:
            #             continue

            # 方法3：处理iframe中的弹窗
            # if not consent_button_found:
            #     print("在主文档中未找到同意按钮，尝试在iframe中查找...")
            #     # 查找页面中的所有iframe
            #     iframes = driver.find_elements(By.TAG_NAME, "iframe")
            #     print(f"找到 {len(iframes)} 个iframe")
            #
            #     # 遍历所有iframe
            #     for i, iframe in enumerate(iframes):
            #         try:
            #             print(f"切换到iframe {i+1}/{len(iframes)}")
            #             driver.switch_to.frame(iframe)
            #
            #             # 在iframe中查找同意按钮
            #             for selector_type, selector in consent_button_selectors:
            #                 try:
            #                     iframe_consent_button = WebDriverWait(driver, 2).until(
            #                         EC.element_to_be_clickable((selector_type, selector))
            #                     )
            #                     if iframe_consent_button:
            #                         print(f"在iframe中找到同意按钮: {iframe_consent_button.text}")
            #                         # 使用JavaScript点击
            #                         driver.execute_script("arguments[0].click();", iframe_consent_button)
            #                         print("已点击iframe中的同意按钮")
            #                         random_sleep(2, 4)  # 等待弹窗关闭
            #                         consent_button_found = True
            #                         break
            #                 except Exception:
            #                     continue
            #
            #             # 如果在当前iframe中找到并点击了按钮，跳出循环
            #             if consent_button_found:
            #                 break
            #
            #             # 切回主文档，准备检查下一个iframe
            #             driver.switch_to.default_content()
            #         except Exception as e:
            #             print(f"处理iframe {i+1} 时出错: {e}")
            #             # 确保切回主文档
            #             driver.switch_to.default_content()
            #             continue
            #
            #     # 确保最后切回主文档
            #     driver.switch_to.default_content()

            # 如果所有方法都未找到同意按钮，尝试按ESC键关闭弹窗
            # if not consent_button_found:
            #     print("未找到任何同意按钮，尝试按ESC键关闭弹窗...")
            #     from selenium.webdriver.common.keys import Keys
            #     webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            #     random_sleep(1, 2)
        except Exception as e:
            print(f"处理服务条款弹窗时出错: {e}")

        error_keywords = ["Connection errored out."]
        page_source = driver.page_source
        found_error_keyword = any(keyword in page_source for keyword in error_keywords)

        if found_error_keyword:
            driver.refresh()
            random_sleep(5, 8)
        else:
            random_sleep(2, 5)

        # 截图记录当前页面状态
        # driver.save_screenshot("after_consent_screenshot.png")
        # print("已保存同意弹窗处理后的页面截图")
        # random_sleep(5, 8)
        # 随机滚动页面，模拟人类浏览行为
        # print("模拟页面浏览行为...")
        # for _ in range(random.randint(2, 5)):
        #     scroll_height = random.randint(100, 700)
        #     driver.execute_script(f"window.scrollBy(0, {scroll_height});")
        #     random_sleep(0, 1)
        retry = 1
        succeed = False
        while retry < 4:
            if clickDirectChat() is True:
                random_sleep(1, 2)
                if clickModel() is True:
                    succeed = True
                    break

            print(f"进行重试: 第{retry}次")
            driver.refresh()
            random_sleep(5, 8)
            retry += 1
        if succeed is False:
            return

        # 设置模型参数
        setParameters()

        # 定位到输入框：//*[@id="input_box"]/div/label/div/textarea
        print("正在定位输入框...")
        try:
            # 等待输入框加载
            input_selectors = [
                # (By.CSS_SELECTOR, "#input_box > div > label > div > textarea"),
                # (By.XPATH, '//*[@id="input_box"]/div/label/div/textarea'),
                (By.CSS_SELECTOR, '#input_box > div > label > div > textarea"'),
                # (By.XPATH, "/html/body/gradio-app/div/div/div[1]/div/div/div/div[4]/div/div[4]/div[2]/div/div/label/div/textarea"),
                # (By.XPATH, "//textarea"),
                # (By.CSS_SELECTOR, "textarea"),
                # (By.CSS_SELECTOR, ".input-box textarea"),
                # (By.CSS_SELECTOR, "[placeholder*='message']"),
                # (By.CSS_SELECTOR, "[placeholder*='输入']"),
            ]

            wait = WebDriverWait(driver, 10)
            input_box = wait.until(
                EC.visibility_of_element_located(
                    (By.XPATH, '//*[@id="component-156"]/div[2]/div/div/label/div/textarea')))

            # input_box = None
            # for selector_type, selector in input_selectors:
            #     try:
            #         print(f"尝试使用选择器: {selector}")
            #         input_box = WebDriverWait(driver, 10).until(
            #             EC.element_to_be_clickable((selector_type, selector))
            #         )
            #         if input_box:
            #             print(f"找到输入框")
            #             break
            #     except Exception as e:
            #         print(f"使用选择器 {selector} 查找输入框失败: {e}")
            #         continue

            if input_box:
                # 滚动到输入框位置
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_box)
                random_sleep(0, 1)

                # 点击输入框
                try:
                    input_box.click()
                    print("成功点击输入框")
                except Exception as click_error:
                    print(f"直接点击输入框失败: {click_error}，尝试JavaScript点击")
                    driver.execute_script("arguments[0].click();", input_box)
                    print("使用JavaScript点击输入框")

                # 输入消息
                print(f"正在输入消息: {message}")

                # 清空输入框
                input_box.clear()

                # 使用不同的输入方法
                try:
                    # 方法1: 直接发送键
                    input_box.send_keys(message)
                    print("使用send_keys方法输入消息")
                except Exception as input_error:
                    print(f"直接输入消息失败: {input_error}，尝试JavaScript输入")
                    # 方法2: 使用JavaScript设置值
                    driver.execute_script("arguments[0].value = arguments[1];", input_box, message)
                    print("使用JavaScript输入消息")

                random_sleep(0, 1)

                # # 查找发送按钮
                # print("正在查找发送按钮...")
                # send_button_selectors = [
                #     (By.XPATH, "//button[contains(@class, 'send') or contains(@class, 'submit')]"),
                #     (By.XPATH, "//button[contains(text(), '发送') or contains(text(), 'Send')]"),
                #     (By.CSS_SELECTOR, "button.send-button"),
                #     (By.CSS_SELECTOR, "button[type='submit']"),
                #     (By.XPATH, "//div[contains(@class, 'send-button')]"),
                # ]

                # send_button = None
                # for selector_type, selector in send_button_selectors:
                #     try:
                #         send_button = WebDriverWait(driver, 5).until(
                #             EC.element_to_be_clickable((selector_type, selector))
                #         )
                #         if send_button:
                #             print(f"找到发送按钮")
                #             break
                #     except Exception as e:
                #         print(f"使用选择器 {selector} 查找发送按钮失败")
                #         continue

                # # 如果找到发送按钮，点击它
                # if send_button:
                #     # 滚动到按钮位置
                #     driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", send_button)
                #     random_sleep(0, 1)

                #     # 点击发送按钮
                #     try:
                #         send_button.click()
                #         print("成功点击发送按钮")
                #     except Exception as click_error:
                #         print(f"直接点击发送按钮失败: {click_error}，尝试JavaScript点击")
                #         driver.execute_script("arguments[0].click();", send_button)
                #         print("使用JavaScript点击发送按钮")
                # else:
                #     # 如果找不到发送按钮，尝试按回车键发送
                #     print("未找到发送按钮，尝试按回车键发送...")
                #     from selenium.webdriver.common.keys import Keys
                #     try:
                #         input_box.send_keys(Keys.RETURN)
                #         print("使用回车键发送消息")
                #     except Exception as enter_error:
                #         print(f"使用回车键发送失败: {enter_error}")
                #         # 尝试使用JavaScript模拟回车键
                #         try:
                #             driver.execute_script("""
                #             var keyEvent = new KeyboardEvent('keydown', {
                #                 key: 'Enter',
                #                 code: 'Enter',
                #                 keyCode: 13,
                #                 which: 13,
                #                 bubbles: true
                #             });
                #             arguments[0].dispatchEvent(keyEvent);
                #             """, input_box)
                #             print("使用JavaScript模拟回车键发送消息")
                #         except Exception as js_enter_error:
                #             print(f"使用JavaScript模拟回车键失败: {js_enter_error}")
                #
                # # 等待AI回复
                # print("等待AI回复...")
                # try:
                #     # 尝试定位回复区域
                #     response_selectors = [
                #         (By.XPATH, "//div[contains(@class, 'message') and contains(@class, 'bot')]"),
                #         (By.XPATH, "//div[contains(@class, 'assistant')]"),
                #         (By.XPATH, "//div[contains(@class, 'response')]"),
                #         (By.XPATH, "//div[contains(@class, 'chat-message') and not(contains(@class, 'user'))]"),
                #     ]

                #     # 等待较长时间，因为AI生成回复可能需要一些时间
                #     response_element = None
                #     for selector_type, selector in response_selectors:
                #         try:
                #             response_element = WebDriverWait(driver, 30).until(
                #                 EC.presence_of_element_located((selector_type, selector))
                #             )
                #             if response_element:
                #                 print(f"找到AI回复元素")
                #                 break
                #         except Exception as e:
                #             print(f"使用选择器 {selector} 查找AI回复失败")
                #             continue

                #     if response_element:
                #         # 等待AI回复完成（可能需要一些时间）
                #         print("等待AI回复完成...")
                #         # 等待一段时间，让AI有足够时间生成完整回复
                #         random_sleep(5, 10)

                #         # 获取回复内容
                #         response_text = response_element.text
                #         print(f"AI回复: {response_text[:100]}..." if len(response_text) > 100 else f"AI回复: {response_text}")

                #         # 保存回复到文件
                #         try:
                #             with open("ai_response.txt", "w", encoding="utf-8") as f:
                #                 f.write(response_text)
                #             print("已保存AI回复到ai_response.txt")
                #         except Exception as save_error:
                #             print(f"保存AI回复时出错: {save_error}")
                #     else:
                #         print("未能找到AI回复元素")
                #         driver.save_screenshot("no_response.png")
                #         print("已保存无回复截图到no_response.png")
                # except Exception as response_error:
                #     print(f"等待AI回复时出错: {response_error}")
                #     driver.save_screenshot("response_error.png")
                #     print("已保存回复错误截图到response_error.png")
            else:
                print("无法找到输入框")
                driver.save_screenshot("no_input_box.png")
                print("已保存无输入框截图到no_input_box.png")
        except Exception as e:
            print(f"处理输入框时出错: {e}")
            driver.save_screenshot("input_error.png")
            print("已保存输入错误截图到input_error.png")

        driver.execute_script(f"window.scrollBy(0, {-350});")
    except Exception as e:
        print("发生错误: {}".format(e))
    finally:
        # # 保存最终页面截图和源码
        # try:
        #     driver.save_screenshot("final_state.png")
        #     print("已保存最终状态截图到final_state.png")
        #
        #     with open("final_page_source.html", "w", encoding="utf-8") as f:
        #         f.write(driver.page_source)
        #     print("已保存最终页面源码到final_page_source.html")
        # except Exception as final_error:
        #     print(f"保存最终状态时出错: {final_error}")

        # 保持浏览器窗口打开
        if succeed is True:
            print("脚本执行完成，浏览器窗口将保持打开状态...")
        # 如果需要保持浏览器窗口打开，可以使用以下方法之一：

        # 方法1：等待用户输入以关闭浏览器
        input("按回车键关闭浏览器...")

        # 方法2：或者使用无限循环保持脚本运行（可以通过Ctrl+C终止）
        # try:
        #     while True:
        #         time.sleep(60)  # 每分钟检查一次
        #         print("浏览器窗口保持打开中...")
        # except KeyboardInterrupt:
        #     print("检测到用户中断，关闭浏览器...")

        # 最后关闭浏览器
        driver.quit()


def clickModel():
    print("正在选择模型...")
    try:
        # 首先尝试点击下拉框以展开选项
        print("尝试点击下拉框以展开选项...")
        dropdown_selectors = [
            # 更精确的选择器 - 基于页面分析
            (By.CSS_SELECTOR, "#component-148 > div.svelte-1sk0pyu > div"),
            (By.XPATH, '//*[@id="component-148"]/div[2]/div'),
            # # 更通用的选择器 - 基于Gradio UI框架
            # (By.CSS_SELECTOR, ".gradio-dropdown"),
            # (By.CSS_SELECTOR, ".gradio-dropdown .svelte-1biyotc"),
            # (By.CSS_SELECTOR, ".gradio-dropdown button"),
            # # 基于属性的选择器
            # (By.XPATH, "//div[contains(@class, 'dropdown') or contains(@class, 'select')]"),
            # (By.XPATH, "//button[contains(@aria-haspopup, 'listbox')]"),
            # (By.XPATH, "//div[contains(@role, 'listbox') or contains(@role, 'combobox')]"),
            # # 尝试查找任何可能的下拉框元素
            # (By.XPATH, "//div[contains(@class, 'gradio')]/div[contains(@class, 'dropdown')]"),
            # (By.XPATH, "//select"),
            # (By.XPATH, "//div[contains(@class, 'select')]"),
        ]

        dropdown_clicked = False
        for selector_type, selector in dropdown_selectors:
            try:
                print(f"尝试使用选择器: {selector}")
                # 增加等待时间，确保元素完全加载
                dropdown = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if dropdown:
                    print(f"找到下拉框元素: {dropdown.get_attribute('outerHTML')[:100]}...")
                    # 先滚动到元素位置
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", dropdown)
                    random_sleep(0, 1)  # 等待滚动完成

                    # 尝试多种点击方法
                    try:
                        # 方法1: 直接点击
                        dropdown.click()
                        print("使用直接点击方法点击下拉框")
                    except Exception as click_error:
                        print(f"直接点击失败: {click_error}，尝试JavaScript点击")
                        # 方法2: 使用JavaScript点击
                        driver.execute_script("arguments[0].click();", dropdown)
                        print("使用JavaScript点击下拉框")

                    print("成功点击下拉框")
                    dropdown_clicked = True
                    # 增加等待时间，确保下拉选项完全加载
                    random_sleep(0, 1)
                    break
            except Exception as e:
                print(f"使用选择器 {selector} 点击下拉框失败: {e}")
                continue

        # 如果无法点击下拉框，尝试直接查找模型选项
        if not dropdown_clicked:
            print("无法点击下拉框，尝试直接查找模型选项...")
            return False
            # driver.save_screenshot("dropdown_error.png")
            # print("已保存下拉框错误截图到dropdown_error.png")

        # 等待下拉选项加载
        # print("等待下拉选项加载...")
        # random_sleep(0, 1)

        # 增加等待时间，确保下拉菜单完全展开
        print("等待下拉菜单完全展开...")
        # random_sleep(0, 1)

        # 尝试多种定位方式查找模型选项
        print("尝试查找模型选项...")
        model_selectors = [
            # 精确匹配模型名称的选择器
            # (By.XPATH, "//div[text()='claude-3-7-sonnet-20250219']"),
            (By.XPATH, "//li[text()='claude-3-7-sonnet-20250219']"),
            # (By.XPATH, "//option[text()='claude-3-7-sonnet-20250219']"),
            # # 包含模型名称的选择器
            # (By.XPATH, "//div[contains(text(), 'claude-3-7-sonnet-20250219')]"),
            # (By.XPATH, "//div[contains(., 'claude-3-7-sonnet-20250219')]"),
            # (By.XPATH, "//li[contains(text(), 'claude-3-7-sonnet-20250219')]"),
            # (By.XPATH, "//option[contains(text(), 'claude-3-7-sonnet-20250219')]"),
            # # 基于Gradio UI框架的选择器
            # (By.CSS_SELECTOR, ".gradio-dropdown .item[data-value*='claude']"),
            # (By.CSS_SELECTOR, ".svelte-1biyotc .svelte-1gqcvjp:contains('claude')"),
            # # 尝试在下拉菜单中查找
            # (By.XPATH, "//div[contains(@class, 'dropdown-menu') or contains(@class, 'select-menu')]//div[contains(text(), 'claude')]"),
            # (By.XPATH, "//ul[contains(@class, 'dropdown-menu') or contains(@class, 'select-menu')]//li[contains(text(), 'claude')]"),
            # # 尝试部分匹配
            # (By.XPATH, "//div[contains(text(), 'claude')]"),
            # (By.XPATH, "//div[contains(text(), 'sonnet')]"),
            # # 通用选择器
            # (By.CSS_SELECTOR, "[data-value*='claude'], [value*='claude']"),
            # (By.CSS_SELECTOR, ".gradio-dropdown .item"),
            # (By.XPATH, "//div[contains(@class, 'dropdown-option')]"),
        ]

        model_option = None
        for selector_type, selector in model_selectors:
            try:
                print(f"尝试使用选择器查找模型: {selector}")
                # 增加等待时间，确保元素完全加载
                model_option = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if model_option:
                    print(f"找到模型选项: {model_option.text if model_option.text else '(无文本内容)'}")
                    # 检查是否包含目标模型名称
                    if 'claude' in model_option.text.lower() or 'sonnet' in model_option.text.lower():
                        print(f"确认找到目标模型: {model_option.text}")
                        break
                    else:
                        print(f"找到的选项不是目标模型，继续查找...")
                        model_option = None
                        continue
            except Exception as e:
                print(f"使用选择器 {selector} 查找模型失败: {e}")
                continue

        if model_option:
            # 先滚动到元素位置
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", model_option)
            # random_sleep(0, 1)  # 等待滚动完成

            # 尝试多种点击方法
            try:
                # 方法1: 直接点击
                model_option.click()
                print("使用直接点击方法选择模型")
            except Exception as click_error:
                print(f"直接点击模型失败: {click_error}，尝试JavaScript点击")
                # 方法2: 使用JavaScript点击
                driver.execute_script("arguments[0].click();", model_option)
                print("使用JavaScript点击选择模型")

            print("成功选择模型")
            return True
            # 保存成功截图
            # random_sleep(0, 1)  # 增加等待时间，确保选择完成
            # driver.save_screenshot("model_success.png")
            # print("已保存模型选择成功截图到model_success.png")
        # else:
        # print("无法找到模型选项，尝试使用JavaScript直接选择...")
        # # 尝试使用更高级的JavaScript方法查找和点击下拉框
        # try:
        #     print("使用高级JavaScript方法查找和点击下拉框...")
        #     dropdown_script = """
        #             // 尝试多种方式查找下拉框
        #             let dropdownSelectors = [
        #                 "#component-148 > div.svelte-1sk0pyu > div",
        #                 ".gradio-dropdown",
        #                 ".gradio-dropdown button",
        #                 "[aria-haspopup='listbox']",
        #                 "select",
        #                 "div.dropdown",
        #                 "div.select"
        #             ];
        #
        #             // 尝试每个选择器
        #             for (let selector of dropdownSelectors) {
        #                 const dropdown = document.querySelector(selector);
        #                 if (dropdown) {
        #                     // 尝试滚动到元素位置
        #                     dropdown.scrollIntoView({block: 'center'});
        #                     // 等待一小段时间后点击
        #                     setTimeout(() => {
        #                         dropdown.click();
        #                     }, 500);
        #                     return {success: true, selector: selector};
        #                 }
        #             }
        #
        #             // 如果没有找到任何匹配的选择器，尝试查找任何可能的下拉框元素
        #             const possibleDropdowns = Array.from(document.querySelectorAll('div, button')).filter(el => {
        #                 const text = el.textContent.toLowerCase();
        #                 const classes = el.className.toLowerCase();
        #                 return classes.includes('dropdown') || classes.includes('select') ||
        #                        el.hasAttribute('aria-haspopup') || el.tagName === 'SELECT';
        #             });
        #
        #             if (possibleDropdowns.length > 0) {
        #                 possibleDropdowns[0].scrollIntoView({block: 'center'});
        #                 setTimeout(() => {
        #                     possibleDropdowns[0].click();
        #                 }, 500);
        #                 return {success: true, selector: 'custom-found'};
        #             }
        #
        #             return {success: false};
        #             """
        #     dropdown_result = driver.execute_script(dropdown_script)
        #     if dropdown_result.get('success'):
        #         print(f"成功使用JavaScript找到并点击下拉框，使用选择器: {dropdown_result.get('selector')}")
        #         dropdown_clicked = True
        #         # 等待下拉选项加载
        #         print("等待下拉选项加载...")
        #         random_sleep(0, 1)  # 增加等待时间，确保下拉选项完全加载
        #
        #         # 尝试查找并点击指定模型
        #         model_script = """
        #                 // 等待一段时间确保下拉菜单已完全展开
        #                 return new Promise(resolve => {
        #                     setTimeout(() => {
        #                         // 尝试多种方式查找模型选项
        #                         // 1. 首先查找可能的下拉菜单容器
        #                         const dropdownContainers = [
        #                             document.querySelector('.gradio-dropdown .dropdown-menu'),
        #                             document.querySelector('.svelte-1biyotc'),
        #                             document.querySelector('[role="listbox"]'),
        #                             document.querySelector('.dropdown-content'),
        #                             document.querySelector('.select-menu'),
        #                             document.body // 如果找不到特定容器，在整个页面中查找
        #                         ].filter(Boolean);
        #
        #                         let modelOption = null;
        #                         let searchResult = {found: false, element: null, method: ''};
        #
        #                         // 2. 在每个容器中查找模型选项
        #                         for (const container of dropdownContainers) {
        #                             if (searchResult.found) break;
        #
        #                             // 获取容器中的所有可能选项
        #                             const options = Array.from(container.querySelectorAll('div, li, option, span, button'));
        #
        #                             // 首先尝试精确匹配
        #                             modelOption = options.find(el =>
        #                                 el.textContent.includes('claude-3-7-sonnet-20250219')
        #                             );
        #
        #                             if (modelOption) {
        #                                 searchResult = {found: true, element: modelOption, method: '精确匹配'};
        #                                 break;
        #                             }
        #
        #                             // 如果没找到，尝试部分匹配
        #                             modelOption = options.find(el =>
        #                                 el.textContent.toLowerCase().includes('claude') &&
        #                                 el.textContent.toLowerCase().includes('sonnet')
        #                             );
        #
        #                             if (modelOption) {
        #                                 searchResult = {found: true, element: modelOption, method: '部分匹配(claude+sonnet)'};
        #                                 break;
        #                             }
        #
        #                             // 如果还没找到，尝试只匹配claude
        #                             modelOption = options.find(el =>
        #                                 el.textContent.toLowerCase().includes('claude')
        #                             );
        #
        #                             if (modelOption) {
        #                                 searchResult = {found: true, element: modelOption, method: '部分匹配(claude)'};
        #                                 break;
        #                             }
        #                         }
        #
        #                         // 3. 如果找到了模型选项，尝试点击它
        #                         if (searchResult.found && searchResult.element) {
        #                             // 滚动到元素位置
        #                             searchResult.element.scrollIntoView({block: 'center'});
        #
        #                             // 等待一小段时间后点击
        #                             setTimeout(() => {
        #                                 try {
        #                                     // 尝试直接点击
        #                                     searchResult.element.click();
        #                                 } catch (e) {
        #                                     // 如果直接点击失败，尝试模拟点击事件
        #                                     const event = new MouseEvent('click', {
        #                                         bubbles: true,
        #                                         cancelable: true,
        #                                         view: window
        #                                     });
        #                                     searchResult.element.dispatchEvent(event);
        #                                 }
        #
        #                                 // 返回结果
        #                                 resolve({
        #                                     success: true,
        #                                     method: searchResult.method,
        #                                     text: searchResult.element.textContent
        #                                 });
        #                             }, 500);
        #                         } else {
        #                             // 如果没有找到任何匹配的选项，返回失败
        #                             resolve({success: false});
        #                         }
        #                     }, 1000); // 等待1秒确保下拉菜单已完全展开
        #                 });
        #                 """
        #
        #         # 执行JavaScript并等待Promise解析
        #         model_result = driver.execute_script(f"return {model_script}")
        #
        #         # 等待JavaScript执行完成
        #         random_sleep(0, 1)
        #
        #         if model_result and model_result.get('success'):
        #             print(
        #                 f"成功使用JavaScript选择模型，方法: {model_result.get('method')}，选择的文本: {model_result.get('text')}")
        #             random_sleep(0, 1)
        #             driver.save_screenshot("js_model_success.png")
        #             print("已保存JavaScript选择模型成功截图到js_model_success.png")
        #         else:
        #             print("JavaScript无法选择模型，尝试列出所有可能的选项...")
        #             # 尝试查找所有可能的下拉选项并打印出来
        #             options_script = """
        #                     return Array.from(document.querySelectorAll('div, li, option')).filter(el =>
        #                         el.textContent.toLowerCase().includes('claude') ||
        #                         el.textContent.toLowerCase().includes('sonnet')
        #                     ).map(el => ({text: el.textContent, html: el.outerHTML.substring(0, 100)}));
        #                     """
        #             options = driver.execute_script(options_script)
        #             if options and len(options) > 0:
        #                 print(f"找到 {len(options)} 个可能的选项:")
        #                 for i, opt in enumerate(options[:5]):  # 只打印前5个
        #                     print(f"选项 {i + 1}: {opt}")
        #
        #                 # 尝试点击第一个包含claude的选项
        #                 click_script = """
        #                         const elements = Array.from(document.querySelectorAll('div, li, option')).filter(el =>
        #                             el.textContent.toLowerCase().includes('claude') ||
        #                             el.textContent.toLowerCase().includes('sonnet')
        #                         );
        #                         if (elements.length > 0) {
        #                             elements[0].click();
        #                             return true;
        #                         }
        #                         return false;
        #                         """
        #                 clicked = driver.execute_script(click_script)
        #                 if clicked:
        #                     print("使用JavaScript成功点击了一个可能的模型选项")
        #                     random_sleep(0, 1)
        #                     driver.save_screenshot("js_model_success.png")
        #                     print("已保存JavaScript选择模型成功截图到js_model_success.png")
        #                 else:
        #                     print("JavaScript无法点击任何选项")
        #             else:
        #                 print("未找到任何可能的选项")
        #         return True
        #     else:
        #         print("无法使用CSS选择器点击下拉框")
        # except Exception as js_error:
        #     print(f"使用JavaScript选择模型时出错: {js_error}")
        #
        # # 保存截图以便分析
        # # driver.save_screenshot("model_screenshot.png")
        # # print("已保存模型选择页面截图到model_screenshot.png")
        #
        # # 尝试获取页面源码以便分析
        # try:
        #     with open("page_source.html", "w", encoding="utf-8") as f:
        #         f.write(driver.page_source)
        #     print("已保存页面源码到page_source.html")
        # except Exception as src_error:
        #     print(f"保存页面源码时出错: {src_error}")

    except Exception as e:
        print(f"选择模型时出错: {e}")
        # driver.save_screenshot("model_error_screenshot.png")
        # print("已保存模型选择错误截图到model_error_screenshot.png")
        #
        # # 尝试获取页面源码以便分析错误
        # try:
        #     with open("error_page_source.html", "w", encoding="utf-8") as f:
        #         f.write(driver.page_source)
        #     print("已保存错误页面源码到error_page_source.html")
        # except Exception as src_error:
        #     print(f"保存错误页面源码时出错: {src_error}")
    return False


def clickDirectChat():
    # 点击"Direct Chat"按钮
    print("正在点击Direct Chat按钮...")
    try:
        # 尝试多种定位方式
        selectors = [
            (By.XPATH, "//button[contains(text(), 'Direct Chat')]"),
            (By.XPATH, "//button[contains(., 'Direct Chat')]"),
            (By.CSS_SELECTOR, "button:contains('Direct Chat')"),
            (By.LINK_TEXT, "Direct Chat"),
            (By.PARTIAL_LINK_TEXT, "Direct"),
        ]

        direct_chat_button = None
        for selector_type, selector in selectors:
            try:
                direct_chat_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if direct_chat_button:
                    break
            except:
                continue

        if direct_chat_button:
            # 使用JavaScript点击，有时比直接点击更可靠
            driver.execute_script("arguments[0].click();", direct_chat_button)
            print("成功点击Direct Chat按钮")
            return True
        else:
            print("无法找到Direct Chat按钮，尝试截图保存")
            # driver.save_screenshot("page_screenshot.png")
            # print("已保存页面截图到page_screenshot.png")
    except Exception as e:
        print(f"点击Direct Chat按钮时出错: {e}")
        # driver.save_screenshot("error_screenshot.png")
        # print("已保存错误截图到error_screenshot.png")
    return False


def setParameters():
    try:
        dropdown_selectors = [
            (By.CSS_SELECTOR, "#component-169"),
        ]

        dropdown_clicked = False
        for selector_type, selector in dropdown_selectors:
            try:
                dropdown = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if dropdown:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", dropdown)
                    try:
                        dropdown.click()
                    except Exception as click_error:
                        print(f"直接点击失败: {click_error}")
                        driver.execute_script("arguments[0].click();", dropdown)

                    dropdown_clicked = True
                    break
            except Exception as e:
                print(f"{selector} 失败: {e}")
                continue

        if not dropdown_clicked:
            print("无法点击下拉框，尝试直接查找模型选项...")
            return

        random_sleep(1, 2)

        model_selectors = [
            (By.XPATH, '//*[@id="component-170"]/div[2]/div/input'),
        ]
        input_box = None
        for selector_type, selector in model_selectors:
            try:
                print(f"尝试使用选择器: {selector}")
                input_box = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if input_box:
                    print(f"找到输入框")
                    break
            except Exception as e:
                print(f"使用选择器 {selector} 查找输入框失败: {e}")
                continue
        if input_box:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_box)
            try:
                input_box.click()
                print("成功点击输入框")
            except Exception as click_error:
                print(f"直接点击输入框失败: {click_error}，尝试JavaScript点击")
                driver.execute_script("arguments[0].click();", input_box)

            input_box.clear()
            try:
                input_box.send_keys(0.2)
            except Exception as input_error:
                print(input_error)
                driver.execute_script("arguments[0].value = arguments[1];", input_box, 0.2)

        model_selectors = [
            (By.XPATH, '//*[@id="component-172"]/div[2]/div/input'),
        ]
        input_box = None
        for selector_type, selector in model_selectors:
            try:
                print(f"尝试使用选择器: {selector}")
                input_box = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                if input_box:
                    break
            except Exception as e:
                print(f"{selector} 失败: {e}")
                continue

        if input_box:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_box)
            random_sleep(0, 1)
            try:
                input_box.click()
            except Exception as click_error:
                print(click_error)
                driver.execute_script("arguments[0].click();", input_box)

            input_box.clear()
            try:
                input_box.send_keys(4096)
            except Exception as input_error:
                print(input_error)
                driver.execute_script("arguments[0].value = arguments[1];", input_box, 4096)

    except Exception as e:
        print(f"设置模型参数失败: {e}")


if __name__ == "__main__":
    content = ''
    if len(sys.argv) >= 2:
        content = sys.argv[1]
    main(content)

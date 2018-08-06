#!/usr/bin/env python
# coding: utf-8
# Date  : 2018-08-06 11:41:30
# Author: b4zinga
# Email : b4zinga@outlook.com
# Func  :

import os
import time
import base64
import datetime
from dateutil.relativedelta import relativedelta

from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from operfile import Operation
from clustering import FrequencyCluster
from settings import ABOUT, ICO, BGMAP, DailyLength, VirusLiveTime, UnMatchCityNum, LostLocationCities, HelpInfo


RUNNING = False  # global variable, whether pause or not.
INTERVAL = datetime.timedelta(days=1)  # global variable, time interval,  default :1 day


class Virus:
    """Define the basic information of influenza.
    """
    def __init__(self):
        self.VIRUS_LIVE_TIME = IntVar()  # Virus survival time (day)
        self.BASE_DAY_TIME = IntVar()  # Daily corresponding time (seconds)
        self.set_virus_live_time()  
        self.set_base_day_time()

    def set_virus_live_time(self, day=VirusLiveTime):
        self.VIRUS_LIVE_TIME.set(day)

    def set_base_day_time(self, seconds=DailyLength):
        self.BASE_DAY_TIME.set(seconds)

    def get_virus_live_time(self):
        return self.VIRUS_LIVE_TIME.get()

    def get_base_day_time(self):
        return self.BASE_DAY_TIME.get()

    def get_virus_live_time_obj(self):
        return self.VIRUS_LIVE_TIME

    def get_base_day_time_obj(self):
        return self.BASE_DAY_TIME


# TODO: use base64 encode bytes to replace temporary .png file.
class Application(Tk):
    """Basic Map Application
    """
    def __init__(self):
        """Initialization
        """
        super().__init__()
        self.createBg()
        self.createIcon()
        self.map = PhotoImage(file='map.png')
        self.can = self.createMap()

    @staticmethod
    def createBg():
        """Create 'map.png' file 
        """
        map_tmp = open("map.png", "wb+")
        map_tmp.write(base64.b64decode(BGMAP))
        map_tmp.close()

    @staticmethod
    def createIcon():
        """Create 'ico.ico' file
        """
        ico_tmp = open('ico.ico', 'wb+')
        ico_tmp.write(base64.b64decode(ICO))
        ico_tmp.close()

    def createMap(self):
        """Create Map
        """
        self.title('地图程序')

        cv = Canvas(self, width=800, height=496, bg='SkyBlue')
        cv.create_image(400, 248, image=self.map)
        os.remove("map.png")
        os.remove("ico.ico")
        cv.grid(row=0, column=0)
        return cv

    def addFunc(self, label_frame):
        label_frame(self, self.can)



class FunctionMenu:
    def __init__(self, root, canvas):
        """Initialization
        """
        self.root = root
        self.root.resizable(False, False)
        # Close button on the upper right corner
        # self.root.protocol("WM_DELETE_WINDOW",self.ask_quit)  
        self.canvas = canvas
        self.virus = Virus()
        # Definition of some variables
        self.fa_path = StringVar()
        self.city_path = StringVar()
        self.show_city = IntVar()
        self.show_virus_num = IntVar()
        self.evolution_speed = IntVar()
        self.from_now = StringVar()
        self.from_now.set('如：2017-01-01')

        self.createFuncFrame()

    def createFuncFrame(self):
        """Function GUI"""
        menu_bar = Menu(self.root)
        menu_bar.add_command(label='修改默认配置', command=self.confVirus)
        menu_bar.add_command(label='关于', command=self.about)
        self.root.config(menu=menu_bar)  # == self.root['menu'] = menu_bar

        frm_main = LabelFrame(self.root, text='功能')
        frm_main.grid(row=0, column=1)

        frm_info = Frame(frm_main)
        frm_info.grid(row=0, column=1)
        Label(frm_info, text='流感存活时间 ').grid(row=1, column=1)
        Label(frm_info, textvariable=self.virus.get_virus_live_time_obj()).grid(row=1, column=2)
        Label(frm_info, text='天, 每天显示').grid(row=1, column=3)
        Label(frm_info, textvariable=self.virus.get_base_day_time_obj()).grid(row=1, column=4)
        Label(frm_info, text='秒').grid(row=1, column=5)
        Label(frm_info, text='').grid(row=2, column=1)

        frm_file = Frame(frm_main)
        frm_file.grid(row=1, column=1)
        Entry(frm_file, textvariable=self.fa_path, ).grid(row=0, column=0)
        Button(frm_file, text="选择fa文件", command=self.selectFaFile, cursor="hand2").grid(row=0, column=1)
        Entry(frm_file, textvariable=self.city_path, ).grid(row=1, column=0, pady=5)
        Button(frm_file, text="选择city文件", command=self.selectCityFile, cursor="hand2").grid(row=1, column=1, pady=5)

        frm_ctrl = Frame(frm_main)
        frm_ctrl.grid(row=2, column=1)
        Checkbutton(frm_ctrl, text='显示城市名', variable=self.show_city, cursor="hand2").grid(row=0, column=0)
        Checkbutton(frm_ctrl, text='显示流感数量', variable=self.show_virus_num, cursor="hand2").grid(row=0, column=1)

        frm_speed = Frame(frm_main, padx=10)  # -------------the frame of change speed button
        frm_speed.grid(row=3, column=1)
        Button(frm_speed, text=" + ", command=self.faster, cursor="hand2").grid(row=1, column=0)
        Label(frm_speed, text="< 速 度 >").grid(row=1, column=1)
        Button(frm_speed, text=" - ", command=self.slower, cursor="hand2").grid(row=1, column=2)

        frm_jump = Frame(frm_main, padx=10, pady=10)
        frm_jump.grid(row=4, column=1)
        Button(frm_jump, text='<<上年', command=self.lastYear, cursor="hand2").grid(row=0, column=0)
        Button(frm_jump, text='<上月', command=self.lastMonth, cursor="hand2").grid(row=0, column=1)
        Button(frm_jump, text='暂停', command=self.pause).grid(row=0, column=2)
        Button(frm_jump, text='下月>', command=self.nextMonth, cursor="hand2").grid(row=0, column=3)
        Button(frm_jump, text='下年>>', command=self.nextYear, cursor="hand2").grid(row=0, column=4)

        frm_time = Frame(frm_main)
        frm_time.grid(row=5, column=1)
        Entry(frm_time, width=15, textvariable=self.from_now).grid(row=0, column=0)
        self.btn_fn = Button(frm_time, text='从当前时间开始显示', command=self.btnFromNow, cursor="hand2")
        self.btn_fn.grid(row=0, column=1)

        frame_show = Frame(frm_main, padx=10, pady=10)
        frame_show.grid(row=6, column=1)
        self.btn_p = Button(frame_show, text='预处理', command=self.btnPreprocessing, cursor="hand2")
        self.btn_p.grid(row=0, column=0)
        self.btn_s = Button(frame_show, text='开始画图', bg='PeachPuff', command=self.btnStart, cursor="hand2")
        self.btn_s.grid(row=0, column=1)
        Button(frame_show, text='清除全部', bg='PeachPuff', command=self.clearAll, cursor="hand2").grid(row=0, column=2)
        Button(frame_show, text='退  出', command=self.askQuit).grid(row=0, column=3)

        frm_clustering = Frame(frm_main, padx=10)
        frm_clustering.grid(row=7, column=1)
        Button(frm_clustering, text="显示聚类图", command=self.showClustering, width=28).grid(row=0, column=1)

    @staticmethod
    def convertTime(t):
        """Converts the date format into the format like that: 2016/01/02
        """
        t = t.replace('-', '/').replace('.', '/').replace(' ', '/')
        try:
            result = datetime.datetime.strptime(t, "%Y/%m/%d")
        except ValueError:
            messagebox.showwarning('Warning', 'Time Format Error')
            return
        return result

    @staticmethod
    def convertJingWei(j, w):
        """convert the real Latitude and longitude into the location on map
        """
        j = 407 + (378 / 180) * float(j)
        w = 248 - (218 / 90) * float(w)
        return j, w


    def confVirus(self):
        """To configure the live time of influenza and daily length.
        """
        def btnOk():
            self.virus.set_virus_live_time(vlt.get())
            self.virus.set_base_day_time(bdt.get())
            conf.destroy()

        conf = Toplevel(self.root)
        conf.title('配置信息')
        conf.resizable(False, False)
        conf.attributes("-topmost", 1)  # WinSetOnTop. or conf.wm_attributes("-topmost", 1)
        Label(conf, text='\t当前存活时间：').grid(row=1, column=1, pady=15)
        vlt = Entry(conf, textvariable=self.virus.get_virus_live_time_obj(), width=10)
        vlt.grid(row=1, column=2)
        Label(conf, text='天\t').grid(row=1, column=3)
        Label(conf, text='\t每天显示时间：').grid(row=2, column=1)
        bdt = Entry(conf, textvariable=self.virus.get_base_day_time_obj(), width=10)
        bdt.grid(row=2, column=2)
        Label(conf, text='秒\t').grid(row=2, column=3)
        Button(conf, text='确定', command=btnOk).grid(row=3, column=1, padx=10, pady=10)
        Button(conf, text='取消', command=conf.destroy).grid(row=3, column=2)

    @staticmethod
    def about():
        """To show author and software information.
        """
        messagebox.showinfo('关于', ABOUT)

    def selectFaFile(self):
        path_ = askopenfilename(filetypes=(("FA files", "*.fa"), ("All files", "*.*")))
        self.fa_path.set(path_)

    def selectCityFile(self):
        path_ = askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        self.city_path.set(path_)

    def faster(self):
        self.evolution_speed.set(self.evolution_speed.get() - 1)

    def slower(self):
        self.evolution_speed.set(self.evolution_speed.get() + 1)

    def lastYear(self):
        global INTERVAL
        self.canvas.delete("tags")
        INTERVAL = relativedelta(years=-1)

    def lastMonth(self):
        global INTERVAL
        self.canvas.delete("tags")
        INTERVAL = relativedelta(months=-1)

    @staticmethod
    def pause():
        global RUNNING
        RUNNING = True

    def nextMonth(self):
        global INTERVAL
        self.canvas.delete("tags")
        INTERVAL = relativedelta(months=1)

    def nextYear(self):
        global INTERVAL
        self.canvas.delete("tags")
        INTERVAL = relativedelta(years=1)

    def btnFromNow(self):
        """show from current time
        """
        if self.fa_path.get() == '':
            messagebox.showwarning('提示', '请选择fa文件')
            return
        if self.city_path.get() == '':
            messagebox.showwarning('提示', '请选择城市经纬度文件')
            return
        if '如' in self.from_now.get() or self.from_now == '':
            messagebox.showwarning('提示', '请输入开始时间')
            return

        self.btn_p.config(state="disabled")
        self.btn_fn.config(state="disabled")
        self.btn_s.config(state="disabled")

        oper = Operation(self.fa_path.get(), self.city_path.get())
        index, start_time, end_time = oper.getDetailByTime()
        end_time = self.convertTime(end_time)
        now = self.convertTime(self.from_now.get())

        try:
            self.showVirus(index, now, end_time)

            self.btn_fn.config(state='normal')
            self.btn_p.config(state='normal')
            self.btn_s.config(state="normal")
        except TclError:
            # Tcl Error: After destory the main window,
            # btn_fn, btn_p, btn_s all have benn destoryed,
            # so, there is no config methods.
            exit(0)

    def btnPreprocessing(self):
        """Complete the city file
        """

        if self.fa_path.get() == '':
            messagebox.showwarning('提示', '请选择fa文件')
            return
        if self.city_path.get() == '':
            messagebox.showwarning('提示', '请选择城市经纬度文件')
            return

        oper = Operation(self.fa_path.get(), self.city_path.get())
        flag = oper.judgeIntegrity()
        if flag:
            if len(flag) > UnMatchCityNum:
                with open(LostLocationCities, 'w') as file:
                    file.write(HelpInfo)
                    for city in flag:
                        file.write(' '*29+city+'\n')
                file.close()
                messagebox.showinfo('CityName', '请在'+os.getcwd() +'/'+ LostLocationCities +'中查看待添加经纬度信息的城市')
                
            else:
                for city in flag:
                    self.addCity(self.city_path.get(), city)
        else:
            messagebox.showinfo('Information', '城市经纬度文件完整！')
        

    def btnStart(self):
        """Reconstructing the whole outbreak process
        """
        if self.fa_path.get() == '':
            messagebox.showwarning('提示', '请选择fa文件')
            return
        if self.city_path.get() == '':
            messagebox.showwarning('提示', '请选择城市经纬度文件')
            return

        self.btn_p.config(state="disabled")
        self.btn_fn.config(state="disabled")
        self.btn_s.config(state="disabled")

        oper = Operation(self.fa_path.get(), self.city_path.get())
        index, start_time, end_time = oper.getDetailByTime()
        end_time = self.convertTime(end_time)
        start_time = self.convertTime(start_time)

        try:
            self.showVirus(index, start_time, end_time)

            self.btn_fn.config(state='normal')
            self.btn_p.config(state='normal')
            self.btn_s.config(state="normal")
        except TclError:
            # Tcl Error: After destory the main window,
            # btn_fn, btn_p, btn_s all have benn destoryed,
            # so, there is no config methods.
            exit(0)

    def clearAll(self):
        self.canvas.delete("tags")

    def askQuit(self):
        messagebox.showwarning('Warning', '确认退出 ?')
        self.root.destroy()

    def showClustering(self):
        if self.fa_path.get() == '':
            messagebox.showwarning('提示', '请选择fa文件')
            return

        fc = FrequencyCluster(self.fa_path.get())
        fc.show()


    def showVirus(self, _index, _start_time, _end_time):
        global RUNNING
        global INTERVAL
        total_virus_num = {}
        s_t = time.clock()

        while _start_time <= _end_time + datetime.timedelta(days=self.virus.get_virus_live_time()):
            virus_number = {}
            info = str(_start_time.strftime("%Y/%m/%d"))
            self.canvas.create_text(400, 400, text=info, fill='red', font=('微软雅黑', 18), tags=("tags", "time",))
            death_virus = (_start_time - datetime.timedelta(days=self.virus.get_virus_live_time())).strftime("%Y/%m/%d")
            try:
                today_virus = _index[_start_time.strftime("%Y/%m/%d")]
                for virus in today_virus:
                    y, x = virus[2]
                    x, y = self.convertJingWei(x, y)
                    virus_number[(x, y)] = virus[3]
                    total_virus_num[(x, y)] = []
                    self.canvas.create_text(x, y, text='●', fill="red", tags=("tags", info,))
                    if self.show_city.get():  # show city name
                        self.canvas.create_text(x, y + 10, text=virus[0], fill="red", tags=("tags", info,))
            except KeyError:
                pass

            for k, v in total_virus_num.items():
                if k in virus_number.keys():
                    v.append(virus_number[k])
                else:
                    v.append(0)
            if self.show_virus_num.get():
                for k, v in total_virus_num.items():
                    if sum(v):
                        self.canvas.create_text(k[0] + 5, k[1], text=str(sum(v)), fill="red",
                                                tags=("tags", info, "num"))
            self.canvas.update()

            for key, value in total_virus_num.items():
                if len(value) > self.virus.get_virus_live_time():
                    total_virus_num[key] = value[-self.virus.get_virus_live_time():]

            # for k in list(total_virus_num.keys()):
            #     if sum(total_virus_num[k]) == 0:
            #         del(total_virus_num[k])

            # print('------------------------',info)
            # print(total_virus_num)

            if RUNNING:
                messagebox.showinfo('提示', '已暂停，点击“确定”继续！')
                RUNNING = False

            speed = self.virus.get_base_day_time() / 10 + self.evolution_speed.get() / 10
            if speed < 0:
                messagebox.showwarning('Warning', '已经达到最大速度')
                # If the speed reaches the max, speed < 0, then speed = 0
                self.evolution_speed.set(-self.virus.get_base_day_time())
                speed = 0.0
            time.sleep(speed)  # one day
            self.canvas.delete("num")
            self.canvas.delete(str(death_virus))  # Delete the virus that died on that day
            self.canvas.delete("time")  # Delete the time on map

            _start_time = _start_time + INTERVAL  # wen click the next year,next month button
            INTERVAL = datetime.timedelta(days=1)  # Reset for one day

        total_time = time.clock() - s_t
        messagebox.showinfo('Information', 'Total time: ' + str(total_time) + ' s')

    @staticmethod
    def addCity(_city_file, place):
        """make a window to input and save the Latitude and longitude of city.
        """
        def btnOk():
            if jing.get() == '' or wei.get() == '':
                messagebox.showwarning('Warning', '请输入经纬度信息')
                return
            if 180 < int(jing.get()) or int(jing.get()) < -180:
                messagebox.showerror('Error', '经度范围错误，请输入-180~180之间数字')
                return
            if 90 < int(wei.get()) or int(wei.get()) < -90:
                messagebox.showerror('Error', '纬度范围错误，请输入-90~90之间数字')
                return
            info = ' ' + wei.get() + '     ' + jing.get() + '     ' + place + '\n'
            with open(_city_file, 'a') as cf:
                cf.write(info)
                cf.close()
            messagebox.showinfo('Information', '添加成功')
            window.destroy()

        window = Toplevel()
        window.resizable(False, False)
        window.attributes("-topmost")
        window.title("添加城市经纬度信息")
        Label(window, text='请输入{}的经纬度信息'.format(place)).grid(row=1, column=1)
        Label(window, text=place + ' 的经度：').grid(row=2, column=1)
        jing = Entry(window)
        jing.grid(row=2, column=2, padx=5)
        Label(window, text=place + ' 的纬度：').grid(row=3, column=1)
        wei = Entry(window)
        wei.grid(row=3, column=2, pady=10)
        Button(window, text='添加', command=btnOk).grid(row=4, column=1)
        Button(window, text='取消', command=window.destroy).grid(row=4, column=2)


if __name__ == '__main__':
    app = Application()
    app.addFunc(FunctionMenu)
    app.mainloop()



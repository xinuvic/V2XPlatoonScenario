# -*- coding: utf-8 -*-
"""
Created on Sun May 15 11:58:49 2022

@author: CSUST
"""
import numpy as np
import math
import random
import seaborn as sns
from ResourceSelectionInitial import ResourceSelectionInitial
from CountConsecutiveColli import CounterConsecutiveNumber,Delay_list
from Distance import Distance
from RSSI import RSSI
from RSSIratePercent import RSSIratePercent
from CalculateSINR_fading import CalculateSINR_fading
from random import choice
from ErrorDetection import ErrorDetection
import matplotlib.pyplot as plt
import sys
import argparse
import time
import pandas as pd

sns.set_style("whitegrid",{'axes.edgecolor': '.0', 'axes.linewidth': 1.0,'grid.color': '1.0','xtick.color' : '0.0',})
sns.set_style("ticks",{'axes.edgecolor': '.0', 'axes.linewidth': 1.0,'grid.color': '1.0','xtick.color' : '0.0',})
sns.set_context("notebook", font_scale = 2)
colors=[sns.xkcd_rgb['velvet'],sns.xkcd_rgb['twilight blue'],sns.xkcd_rgb['leaf'],sns.xkcd_rgb['orangered'],sns.xkcd_rgb['slate grey'],sns.xkcd_rgb['medium green']]



parser = argparse.ArgumentParser(description=\
                                 '--lvn: leading num in IFT,\
                                 \n--fn: following num in IFT,\
                                 \n--r: running time,\
                                 \n--est: simulation time,\
                                 \n--sst: start sampling time,\
                                 \n--si: scheme index 0(sps)1(fb),\
                                 \n--td: target distance for beacon message,\
                                 \n--db: delay bound,\
                                 \n--fade: fading or not,\
                                 \n--itv: transmission interval of beacon messages,\
                                 \n--inp: inter-distance between platoon vehicles')
                                 
parser.add_argument('--lvn', type=int, default=2)
parser.add_argument('--fn', type=int, default=2)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--est', type=int, default=150)
parser.add_argument('--sst', type=int, default=10)
parser.add_argument('--si', type=int, default=0)
parser.add_argument('--td', type=float, default=200) 
parser.add_argument('--db', type=float, default=500)
parser.add_argument('--fade', type=str, default='off')
parser.add_argument('--itv', type=int, default=100)
parser.add_argument('--inp', type=float, default=10)

def output_dop(runningtime,platoon_index,state_list_r2l2_allruns,maximalTime):
    dop_list=[0 for i in range(0,runningtime)]
    for s in range(0,runningtime):
        accumulate_1_r2l2=[0 for i in range(0,len(platoon_index))]
        sum_fail_r2l2=[0 for i in range(0,len(platoon_index))]
        for i in range(0,len(platoon_index)):
            accum_0_list_r2l2,accumulate_1_r2l2[i]=CounterConsecutiveNumber(state_list_r2l2_allruns[s][i])
            sum_fail_r2l2[i]=sum(x*(x>maximalTime) for x in accum_0_list_r2l2)
        sum_all_r2l2=sum(sum_fail_r2l2)+sum(accumulate_1_r2l2) 
        dop_list[s]=sum(sum_fail_r2l2)/sum_all_r2l2    
        dop_list_average=np.average(dop_list)
        dop_list_std=math.sqrt(sum([(i-dop_list_average)**2 for i in dop_list])/len(dop_list))
        return dop_list_average,dop_list_std

def select_column(a,n):
    return [x[n] for x in a]    


def TargetPlatoonDistance(i,j,IntraPlatoonDistance,LeadingNum,FollowNum,VehicleLength):   
    if i>j:
        return FollowNum*(IntraPlatoonDistance+VehicleLength)+1
    else:
        return LeadingNum*(IntraPlatoonDistance+VehicleLength)+1


    
def SimulationwithSPS(ResourceSelectionini,DesiredDistance,RClist,NumVehicle,StartTime,VehicleLocation,RCrange,platoon_index,non_platoon,platoon_index_ori,Feedback):
    lowerbound=RCrange[0]
    higherbound=RCrange[1]   
    ave_rc=int(np.average(RCrange))
    RSSIEach = [0]*SubchannelNum    
    RSSIEachStatistic=[[] for i in range(0,NumVehicle)]
    sumRSSI = []
    AverageRSSI = []
    PlatoonPacketCollision = 0
    PlatoonPacketCollision_i2j = 0
    PacketCollision = 0
    Platoonalltrans=0
    Platoonalltrans_i2j=0
    alltrans=0
    ResourceSelectionallEachRound = ResourceSelectionini[:]
    ResourceSelectionall = ResourceSelectionini[:]
    state_list_r2l2=[[1 for k in range(0,SimulationTime-StartTime)] for i in range(0,len(platoon_index))]
    # for all observed platoon vehicles, for each receiver, the access conditon list by time
    state_list_i2j=[[[1 for k in range(0,SimulationTime-StartTime)] for j in range(0,LeadingNum+FollowNum)] for i in range(0,len(platoon_index_ori))]
    # for all observed platoon vehicles, for each receiver, the total lost packet numbers
    pc_list_i2j=[[0 for j in range(0,LeadingNum+FollowNum)] for i in range(0,len(platoon_index_ori))]
    alltrans_i2j=[[0 for j in range(0,LeadingNum+FollowNum)] for i in range(0,len(platoon_index_ori))]
    RClist_rechosen=ResourceSelectionini
    change_num=0
    false_num=0
    RC_alltime_allvehicle=[[] for i in range(0,SimulationTime)]
    sum_fail_r2l2=[0 for i in range(0,len(platoon_index))]
    sum_all_r2l2=[0 for i in range(0,len(platoon_index))]
    accumulate_1_r2l2=[0 for i in range(0,len(platoon_index))]
    accum_dd_0and1_list=[[] for i in range(0,len(platoon_index))]
    accum_dd_i2j_list=[[] for i in range(0,LeadingNum+FollowNum)]

    for t in range(1,SimulationTime):
        VehicleLocation=observe_vehicles[t]
        if t%5==0: 
            print('t=',t)
        for i in range(0,NumVehicle):
            RClist[i]=RClist[i]-1
            RSSIEach = RSSI(i,ResourceSelectionall,SubchannelNum,NumVehicle,VehicleLocation,TransmitPower_mw)
            RSSIEachStatistic[i].append(RSSIEach)
            if t<ave_rc:
                sumRSSI = np.sum(RSSIEachStatistic[i],axis=0)
                AverageRSSI = [m/t for m in sumRSSI]
            else:
                sumRSSI = np.sum(RSSIEachStatistic[i][t-ave_rc+1:],axis=0)
                AverageRSSI = [i/ave_rc for i in sumRSSI]
            if RClist[i] == 0:
                RClist[i] = random.randint(lowerbound,higherbound+1)
                if i in platoon_index:
                    RClist_rechosen[i]=RClist[i] 
                p = random.random()
                if p > ProbabilityofPersistance:
                    temp = RSSIratePercent(i,AverageRSSI,ResourceSelectionall,SubchannelNum,0.2)
                    subchselected = choice(temp)
                    ResourceSelection_i = subchselected
                    ResourceSelectionallEachRound[i] = ResourceSelection_i
        ResourceSelectionall = ResourceSelectionallEachRound[:]
        RC_alltime_allvehicle[t]=RClist[:]
        
        if Feedback=='on':
            #print('**************execute the collision check procedure****************')
            FirstIndex=platoon_index[0]
            for i in platoon_index:
                #print('i:',i,'before check, resource is',ResourceSelectionallEachRound[i])
                if t>=3 and (2<=RClist_rechosen[i]-RC_alltime_allvehicle[t][i]<=RCrange[0]):
                    cc,false_alarm=ErrorDetection(i,FirstIndex,PlatoonLen,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,sinr_th,max(LeadingNum,FollowNum),fading,scale_param_omega)
                    if cc==1:   
                        change_num+=1
                        if false_alarm==1:
                            false_num+=1
                        if t<ave_rc:
                            sumRSSI = np.sum(RSSIEachStatistic[i],axis=0)
                            AverageRSSI = [m/t for m in sumRSSI]
                        else:
                            sumRSSI = np.sum(RSSIEachStatistic[i][t-ave_rc+1:],axis=0)
                            AverageRSSI = [i/ave_rc for i in sumRSSI]                        
                        temp = RSSIratePercent(i,AverageRSSI,ResourceSelectionall,SubchannelNum,0.2)
                        subchselected = choice(temp)
                        ResourceSelectionallEachRound[i] = subchselected
                        
            ResourceSelectionall = ResourceSelectionallEachRound[:]
        if t in range(StartTime,SimulationTime):
            for i in platoon_index:
                Platoonalltrans+=1 # four receivers counted together so only counted once for each transmitter i
                for j in platoon_index_ori:# evaluate platoon communication PDR
                    if i == j:
                        continue
                               
                    if -LeadingNum<=(j-i)<=FollowNum and CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,fading,scale_param_omega)<sinr_th:
                        PlatoonPacketCollision+=1 # sum up the conflicted packet num from i to 4 neighbours from t_start to t_end
                        #print(i,'to',j,'packet lost','distance:',Distance(i,j,VehicleLocation))
                        state_list_r2l2[i-platoon_index[0]][t-StartTime]=0
                        break
                    elif -LeadingNum<=(j-i)<=FollowNum and CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,fading,scale_param_omega)>=sinr_th:
                        delay_in_this_period=(ResourceSelectionall[i]//2+1)
                        state_list_r2l2[i-platoon_index[0]][t-StartTime]=delay_in_this_period

            # statistics of i->j transmission collision
            for i in platoon_index_ori:
                for j in platoon_index_ori:# evaluate platoon communication PDR
                    if i == j:
                        continue
                    if -LeadingNum<=(j-i)<=FollowNum: Platoonalltrans_i2j+=1                               
                    if -LeadingNum<=(j-i)<=FollowNum and CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,fading,scale_param_omega)<sinr_th:#if fail
                        PlatoonPacketCollision_i2j+=1 # sum up the conflicted packet num from i to 4 neighbours from t_start to t_end
                        if j<i:
                            state_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum][t-StartTime]=0
                            pc_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum]+=1
                            alltrans_i2j[i-platoon_index_ori[0]][j-i+LeadingNum]+=1
                        if j>i:
                            state_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum-1][t-StartTime]=0
                            pc_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum-1]+=1
                            alltrans_i2j[i-platoon_index_ori[0]][j-i+LeadingNum-1]+=1
                    elif -LeadingNum<=(j-i)<=FollowNum and  CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,fading,scale_param_omega)>=sinr_th:#if success
                        delay_in_this_period=(ResourceSelectionall[i]//2+1)
                        if j<i:
                            state_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum][t-StartTime]=delay_in_this_period
                            alltrans_i2j[i-platoon_index_ori[0]][j-i+LeadingNum]+=1
                        if j>i:
                            state_list_i2j[i-platoon_index_ori[0]][j-i+LeadingNum-1][t-StartTime]=delay_in_this_period
                            alltrans_i2j[i-platoon_index_ori[0]][j-i+LeadingNum-1]+=1


            for i in non_platoon:
                for j in range(0,NumVehicle):# evaluate non-platoon vehicle transmission
                    if i == j:
                        continue                               
                    if Distance(i,j,VehicleLocation)<DesiredDistance:
                        alltrans+=1
                        if CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,TransmitPower_mw,fading,scale_param_omega)<sinr_th:
                            PacketCollision+=1 
    
    for i in range(0,len(platoon_index)):
        accum_0_list_r2l2,accumulate_1_r2l2[i]=CounterConsecutiveNumber(state_list_r2l2[i])
        accum_dd_0and1_list[i]=Delay_list(state_list_r2l2[i], TransmitInterval)
        sum_fail_r2l2[i]=sum(x*(x>maximalTime) for x in accum_0_list_r2l2)

    sum_all_r2l2=sum(sum_fail_r2l2)+sum(accumulate_1_r2l2) 
    dop_r2l2=sum(sum_fail_r2l2)/sum_all_r2l2
    sum_coop_delay=sum(accum_dd_0and1_list,[]) # all delay 

    for i in range(0,len(platoon_index)):
        for j in range(0,LeadingNum+FollowNum):
            accum_dd_i2j_list[j]+=Delay_list(state_list_i2j[i][j], TransmitInterval)
    goodput = Platoonalltrans_i2j-PlatoonPacketCollision_i2j
    print('Platoon_all_trans_i2j:',Platoonalltrans_i2j)
    print('Platoon_Packet_Collision_i2j:',PlatoonPacketCollision_i2j)
    return goodput,1-PlatoonPacketCollision/Platoonalltrans,1-PacketCollision/alltrans,dop_r2l2,state_list_r2l2,sum_coop_delay,accum_dd_i2j_list,false_num, change_num



def run_simu(VehicleLocation,DesiredDistance,RCrange,Feedback,aviod_collision_ini,ras):
    
    PacketDeliveryRatiolist=[[]]*runningtime
    PlatoonPacketDeliveryRatiolist=[[]]*runningtime
    accum_dd_i2j_list_allruns=[[]]*runningtime
    sum_coop_delay_allruns=[[]]*runningtime
    goodput_list=[[]]*runningtime
    false_total=0
    change_total=0

    lowerbound=RCrange[0]
    higherbound=RCrange[1] 
    doplist=[[] for i in range(0,runningtime)]       
    RClist = [random.randint(lowerbound,higherbound) for i in range(0,NumVehicle)] 
    state_list_r2l2_allruns=[[] for i in range(0,runningtime)] 
    
        
    set_of_platoon=np.array(pd.read_csv("set_of_platoon.csv",header=None)).tolist()[0]
    set_of_non_platoon=np.array(pd.read_csv("set_of_non_platoon.csv",header=None)).tolist()[0]
    
    
    for s in range(0,runningtime):
        platoon_index_ori=set_of_platoon
        non_platoon=set_of_non_platoon
        print('simulation round',s+1)
        platoon_index=platoon_index_ori[LeadingNum:-FollowNum]
        if s==0:
            print('ori platoon_index:',platoon_index_ori)
            print('observe platoon_index:',platoon_index)
        ResourceSelectionini= ResourceSelectionInitial(NumVehicle,SubchannelNum,aviod_collision_ini)
        goodput,PlatoonPDR,PDR,dop,state_list_r2l2,sum_coop_delay,accum_dd_i2j_list,false_num, change_num = SimulationwithSPS(ResourceSelectionini,DesiredDistance,RClist,NumVehicle,StartTime,VehicleLocation,RCrange,platoon_index,non_platoon,platoon_index_ori,Feedback)        

        sum_coop_delay_allruns[s]=sum_coop_delay   
        PacketDeliveryRatiolist[s]=PDR
        PlatoonPacketDeliveryRatiolist[s]=PlatoonPDR
        doplist[s]=dop
        state_list_r2l2_allruns[s]=state_list_r2l2
        accum_dd_i2j_list_allruns[s]=accum_dd_i2j_list
        goodput_list[s] = goodput
        
        false_total += false_num
        change_total += change_num
        
    PDR_ave = sum(PacketDeliveryRatiolist)/float(len(PacketDeliveryRatiolist))
    PlatoonPDR_ave = sum(PlatoonPacketDeliveryRatiolist)/float(len(PlatoonPacketDeliveryRatiolist))
    dop_ave = np.average(doplist)
    std_dop = math.sqrt(sum([(i-dop_ave)**2 for i in doplist])/len(doplist))
    StdPC = math.sqrt(sum([(i-PDR_ave)**2 for i in PacketDeliveryRatiolist])/len(PacketDeliveryRatiolist))
    PlatoonStdPC = math.sqrt(sum([(i-PlatoonPDR_ave)**2 for i in PlatoonPacketDeliveryRatiolist])/len(PlatoonPacketDeliveryRatiolist))
    print('t from %d'%StartTime,'to %d'%int(SimulationTime))
    
    return goodput_list,PlatoonPDR_ave,PlatoonStdPC,PDR_ave,StdPC,dop_ave,std_dop,state_list_r2l2_allruns,platoon_index,accum_dd_i2j_list_allruns,sum_coop_delay_allruns,false_total,change_total


def autolabel(rects):
    for rect in rects:
        height = round(rect.get_height(),1)
        plt.text(rect.get_x()+rect.get_width()-0.13, 1.01*height, '%s' % float(height),fontsize=25)

def main(leading_num,following_num,running_time,simulation_time,\
         start_time,SchemeIndex,desired_distance,threshold,fade,transmission_interval,intd_platoon):
    global BeaconRate,lane,LeadingNum,FollowNum,fading,scale_param_omega,\
        SubchannelNum,lane,sinr_th,TransmitPower_mw,runningtime,\
            PlatoonLen,VehicleLength,IntraDistance,\
                IntraPlatoonDistance,StartTime,SimulationTime,\
                    ProbabilityofPersistance,TransmitInterval,maximalTime
                    
    BeaconRate = int(1000/transmission_interval)
    vehicle_simu=vehicle_num
    LeadingNum=leading_num
    FollowNum=following_num
    scale_param_omega=1
    fading=fade

    IntraPlatoonDistance=intd_platoon
    
    
    scheme_list=['sps','fb']
    scheme_index=SchemeIndex
    scheme=scheme_list[scheme_index]
    
    if fading=='on':
        fading_indicator='fading_'
    else:
        fading_indicator='no_fading_'
    
    filename='new_'+fading_indicator+scheme+'_t'+str(LeadingNum)+'r'+str(FollowNum)\
        +str(BeaconRate)+'hz_'+'_intp'+str(int(IntraPlatoonDistance))+'_r'+str(running_time)\
            +time.strftime("_%Y-%m-%d-%H-%M-%S", time.localtime())
    
    '''
    f = open('%s.log'%(filename), 'a')
    sys.stdout = f
    '''
    print('********************new run***********************')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('transmitter num:',LeadingNum)
    print('receiver num:',FollowNum)
    print('scheme:',scheme)
    print('fading:',fading)
    
    DesiredDistance=desired_distance # desired transmission range for beacon broadcasting
    
    runningtime = running_time
    StartTime = start_time
    SimulationTime = simulation_time
    sinr_th_db=2.76
    sinr_th=10**(sinr_th_db/10)
    TransmitPowerdBm= 23
    TransmitPower_mw = 10**(TransmitPowerdBm/10)
    RCrange1 = [5,15]
    RCrange2 = [10,30]
    RCrange3 = [25,75]
    
    if BeaconRate==10:RCrange=RCrange1
    if BeaconRate==20:RCrange=RCrange2
    if BeaconRate==50:RCrange=RCrange3
    
    print('beacon rate is',BeaconRate)
    SubchannelNum = int(2*(1000/BeaconRate))  
    TransmitInterval=1000*1/BeaconRate
    maximalTime=np.floor(threshold/TransmitInterval)
    
    #ProbabilityofPersistance in range [0,0.8]
    ProbabilityofPersistance = 0
    print('Probability of Persistance:',ProbabilityofPersistance)
    #d_th=19.67
    VehicleLength = 4.0 
    
    aviod_collision_ini=False
    if aviod_collision_ini==True:
        print('resource selection initialization without collision')
    else:
        print('resource selection initialization is all random')
    
    plr_list=[]
    plr_std_list=[]
    nonplatoon_pdr_list=[]
    goodput_list=[]
    goodput_list_all=[]
    
    dop_r2l2_list=[]
    std_dop_r2l2_list=[]
    false_total_list=[]
    change_total_list=[]



    for i in range(1):
        print('Intra-Platoon Distance',IntraPlatoonDistance)
        print('Desired Beacon Distance',DesiredDistance)

        global NumVehicle
        NumVehicle = vehicle_num # this is total number of all vehicles    
        PlatoonLen=10
        VehicleLocation=observe_vehicles[0]


        if scheme=='fb':
            print('**************CRR*************')
            goodput_all_run,PlatoonPDR,PlatoonPDRstd,PDR,PDRstd,dop_r2l2,std_dop_r2l2,state_list_r2l2_allruns,platoon_index,accum_dd_i2j,sum_coop_delay,false_total,change_total=run_simu(VehicleLocation,DesiredDistance,RCrange,'on',aviod_collision_ini,'off')
        if scheme=='sps':
            print('**************SPS*************')
            goodput_all_run,PlatoonPDR,PlatoonPDRstd,PDR,PDRstd,dop_r2l2,std_dop_r2l2,state_list_r2l2_allruns,platoon_index,accum_dd_i2j,sum_coop_delay,false_total,change_total=run_simu(VehicleLocation,DesiredDistance,RCrange,'off',aviod_collision_ini,'off')
    
    
        plr_list.append(1-PlatoonPDR)
        plr_std_list.append(PlatoonPDRstd)
        dop_r2l2_list.append(dop_r2l2)
        std_dop_r2l2_list.append(std_dop_r2l2)    
        nonplatoon_pdr_list.append(PDR)
        false_total_list.append(false_total)
        change_total_list.append(change_total)
        goodput_list_all.append(goodput_all_run)
        goodput_list.append(np.average(goodput_all_run))
        
        
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('number of vehicle:',vehicle_num)
        print('plr ',plr_list[-1])
        print('plr_std ',plr_std_list[-1])
        print('dop_r2l2 ',dop_r2l2_list[-1],'std',std_dop_r2l2_list[-1])
        print('non-platoon pdr',nonplatoon_pdr_list[-1])
        print('false_total_list ',false_total_list[-1])
        print('change_total_list ',change_total_list[-1])
        print('goodput_list ',goodput_list[-1])
        print('goodput_list_all ',goodput_list_all[-1])
    
    s=fading_indicator+'_'+str(BeaconRate)+'Hz'\
        +'_'+scheme+'_n'+str(vehicle_simu)\
            +'t'+str(LeadingNum)+'r'+str(FollowNum)\
                +'intp'+str(int(IntraPlatoonDistance))
    filename1=s+'_plr'
    filename2=s+'_dop'
    filename3=s+'_gp'
    
    n=0
    for i in plr_list:
        if n<len(plr_list)-1:
            with open(r'%s.csv'%filename1,'a',encoding='utf8') as name:
                name.write(str(i) + '\n')
            n+=1
        else:
            with open(r'%s.csv'%filename1,'a',encoding='utf8') as name:
                name.write(str(i)+ '\n')
            n+=1

    n=0
    for i in dop_r2l2_list:
        if n<len(dop_r2l2_list)-1:
            with open(r'%s.csv'%filename2,'a',encoding='utf8') as name:
                name.write(str(i) + '\n')
            n+=1
        else:
            with open(r'%s.csv'%filename2,'a',encoding='utf8') as name:
                name.write(str(i)+ '\n')
            n+=1


    n=0
    for i in goodput_list:
        if n<len(goodput_list)-1:
            with open(r'%s.csv'%filename3,'a',encoding='utf8') as name:
                name.write(str(i) + '\n')
            n+=1
        else:
            with open(r'%s.csv'%filename3,'a',encoding='utf8') as name:
                name.write(str(i))
            n+=1

if __name__ == '__main__':
    sumo_data_time=150
    observe_vehicles = [[] for i in range(0,sumo_data_time)]
    data_all=np.array(pd.read_csv("realmap_vehicle_location.csv",header=None)).tolist()
    vehicle_num=int(len(data_all)/sumo_data_time)
    for i in range(0,sumo_data_time):
        observe_vehicles[i]=data_all[int(i*vehicle_num):int((i+1)*vehicle_num)]
    args = parser.parse_args()   
    main(args.lvn,args.fn,args.r,args.est,args.sst,args.si,args.td,args.db,args.fade,args.itv,args.inp)
        



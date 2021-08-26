from os import error, system, path, mkdir
from fractions import Fraction
from functools import reduce
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from typing import Type, TypeVar
from itertools import combinations
from matplotlib.font_manager import FontProperties
from  WConio2 import getch
import ctypes
# import itertools as it

LF_FACESIZE = 32
STD_OUTPUT_HANDLE = -11

class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

class CONSOLE_FONT_INFOEX(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong),
                ("nFont", ctypes.c_ulong),
                ("dwFontSize", COORD),
                ("FontFamily", ctypes.c_uint),
                ("FontWeight", ctypes.c_uint),
                ("FaceName", ctypes.c_wchar * LF_FACESIZE)]

def set_font_style():
    font = CONSOLE_FONT_INFOEX()
    font.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
    font.nFont = 23
    font.dwFontSize.X = 13
    font.dwFontSize.Y = 20
    font.FontFamily = 54
    font.FontWeight = 700
    font.FaceName = "Consolas" #"DejaVu Sans Mono"
    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    ctypes.windll.kernel32.SetCurrentConsoleFontEx(
            handle, ctypes.c_long(False), ctypes.pointer(font))


def accept_any_key():
    print("\nPress any key to continue..........")
    getch()
    
system("color 1f")


class Point:
    def __init__(self, x, y):
        self.x = eval(str(x))
        self.y = eval(str(y))

    @property
    def abscissa(self):
        return self.x
    
    @property
    def ordinate(self):
        return self.y
    
    #@property
    def distanceFrom(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5  
    
    @property
    def quadrant(self):
        if self.x > 0 and self.y > 0:
            return "First"
        elif self.x < 0 and self.y > 0:
            return "Second"
        elif self.x < 0 and self.y < 0:
            return "Third"
        elif self.x > 0 and self.y < 0:
            return "Fourth"
        elif self.x == 0 and self.y == 0:
            return "Origin"
        elif self.x == 0:
            if self.y > 0:
                return "Positve Y"
            else:
                return "Positve Y"
        elif self.y == 0:
            if self.x > 0:
                return "Positve X"
            else:
                return "Positve X"
        else:
            raise ValueError("Cannot find in the Quadrant")
    
    def midPoint(self,point):
        return Point((self.x + point.x)/2, (self.y + point.y)/2)
    
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False
            
    def __str__(self):
        return f"({round(self.x,3)}, {round(self.y,3)})"
    
    def __repr__(self):
        return f"Point({round(self.x,3)}, {round(self.y,3)})"   


class StraightLine:
    
    plotObject = plt
    
    def __init__(self, a=0, b=0, c=0, point1: Point = None, point2 : Point = None, 
        slope: float = None, xIntercept: float = None, yIntercept: float = None,alpha=None,p=None,specialText=None):
        #if point1 is None
        self.isInterseptForm = False
        self.isNormalForm = False
        self.isTwoPointForm = False
        self.isPointSlopeForm = False
        self.isGeneralForm = False
        self.isSlopeInterceptForm = False
        self.isPointInterceptFormX = False
        self.isPointInterceptFormY = False
        self.specialText = specialText
        self.SpecialPoints = tuple()
        if a == 0 and b == 0:
            if slope is None:
                if point1 is None and point2 is None:
                    if xIntercept is None or yIntercept is None:
                        if alpha == None or p == None:
                            raise ValueError("It is impossible to create a straight line")
                        else:
                            self.a = p / math.sin(alpha * math.pi / 180)
                            self.b = p / math.cos(alpha * math.pi / 180)
                            self.c = -(self.a * self.b)
                            self.isNormalForm = True
                    else:
                        self.a = yIntercept
                        self.b = xIntercept
                        self.c = -(xIntercept * yIntercept)
                        self.isInterseptForm = True
                elif point1 is not None and (xIntercept is not None or yIntercept is not None):
                    if xIntercept is not None:
                        self.a = point1.ordinate/(1 - (point1.abscissa / xIntercept))
                        self.b = xIntercept
                        self.c = -(self.a * self.b)
                        self.isPointInterceptFormX = True
                    elif yIntercept is not None:
                        self.a = yIntercept
                        self.b = point1.abscissa/(1 - (point1.ordinate / yIntercept))
                        self.c = -(self.a * self.b)
                        self.isPointInterceptFormY = True
                elif point1 is not None and point2 is not None:
                    try:
                        slope = ((point2.ordinate - point1.ordinate)/(point2.abscissa - point1.abscissa))
                        self.a = slope
                        self.b = -1
                        self.c = point1.ordinate - (slope * point1.abscissa)
                        self.SpecialPoints = (point1,point2)
                        self.isTwoPointForm = True
                    except ZeroDivisionError:
                        self.a = 1
                        self.b = 0
                        self.c = - point1.abscissa
                        self.SpecialPoints = (point1,point2)
                        self.isTwoPointForm = True
            else:
                if point1 is None and point2 is None:
                    if yIntercept is not None:
                            self.a = slope
                            self.b = -1
                            self.c = yIntercept
                            self.isSlopeInterceptForm = True
                    else:
                        raise ValueError("It is impossible to create a straight line")
                elif point1 is not None:
                    self.a = slope
                    self.b = -1
                    self.c = point1.ordinate - (slope * point1.abscissa)
                    self.SpecialPoints = (point1,)
                    self.isPointSlopeForm = True
                elif point2 is not None:
                    self.a = slope
                    self.b = -1
                    self.c = point2.ordinate - (slope * point2.abscissa)
                    self.SpecialPoints = (point2,)         
                    self.isPointSlopeForm = True
        else:
            self.a = a
            self.b = b
            self.c = c
            self.isGeneralForm = True   
    
    
    @classmethod
    def getReadyPlotObject(cls,style):
        style.use('ggplot')
        figu = cls.plotObject.figure(figsize=(8,8),dpi=100, facecolor='w', edgecolor='k')
        # fig = figu.add_subplot(111)
        # figu.subplots_adjust(left=0.06, right=0.97)
        cls.font = FontProperties()
        cls.font.set_family('monospace')
        cls.font.set_name('Courier New')
        # cls.font.set_style('italic')
        cls.font.set_weight('heavy')
        cls.font.set_size('large')
        
        # cls.plotObject.scatter(0,0,s=100,marker='o',cmap="Dark2_r",c=[0])
        cls.plotObject.tick_params(axis='both', which='major', labelsize=9)
        # cls.plotObject.annotate(f"(0,0)", # this is the text
        #                 (0,0), # this is the point to label
        #                 textcoords="offset points", # how to position the text
        #                 xytext=(30,35), # distance from text to points (x,y)
        #                 ha='center',
        #                 arrowprops = dict(arrowstyle="fancy",color="r")) # horizontal alignment can be left, right or center
        cls.plotObject.ylabel('Y axis',fontproperties=cls.font)
        cls.plotObject.xlabel('X axis',fontproperties=cls.font)
        # cls.plotObject.colorbar().set_levels('Satisfaction')
        
        
        
    def linePlot(self,plotObject,xyText,plotColor,max_distance):
        xText, yText = xyText
        font = StraightLine.font
        # font.set_family('monospace')
        # font.set_weight('heavy')
        # font.set_size('large')


        for pt in self.SpecialPoints:
            # font = FontProperties()
            # font.set_family('monospace')
            # font.set_weight('heavy')
            # font.set_size('large')
            StraightLine.plotObject.scatter(pt.x,pt.y,s=100,marker='o',cmap="Dark2_r",c=[pt.y])
            StraightLine.plotObject.annotate(f"({round(pt.x,2)},{round(pt.y,2)})", # this is the text
                    (pt.x,pt.y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, -40), # distance from text to points (x,y)
                    ha='center',
                    fontproperties= font,
                    arrowprops = dict(arrowstyle="fancy",color="red")) # horizontal alignment can be left, right or center

        
        if self.a != 0 and self.b != 0:
            # if len(self.SpecialPoints) > 0:
            maximum = max(abs(eval(self.xIntercept)),max_distance,
            max(tuple(map(lambda point: abs(point.x),self.SpecialPoints))) if len(self.SpecialPoints) > 0 else 0)
            x = np.linspace(-maximum-1,maximum+1,100)
            # x = np.linspace(-abs(eval(self.xIntercept))-10,abs(eval(self.xIntercept))+10,100)
            y =  eval(self.slopeInterceptFormat.replace('y = ','').replace(' x','*x')) # np.linspace(-maximum-10,maximum+10,100)
            plt.plot(x,eval(self.slopeInterceptFormat.replace('y = ','').replace(' x','*x')),plotColor,
                        label=
                        (self.specialText + "\n" if self.specialText is not None else "" )+
                        self.slopeInterceptFormat + '\n' 
                        #+self.interseptFormat + '\n' +
                        + 'X-intercept: ' + f"{round(eval(self.xIntercept),3)}" + '\n' 
                        + 'Y-intercept: ' + f"{round(eval(self.yIntercept),3)}" + '\n'
                        , linewidth=1)
            plt.plot(x,0*x,'r',linewidth=1) # label='x-axis',
            plt.plot(0*y,y,'r',linewidth=1) # label='y-axis',
            plt.scatter([eval(self.xIntercept),0],[0,eval(self.yIntercept)],s=100,marker='o',cmap="Dark2_r",c=[0,eval(self.yIntercept)])
            if eval(self.xIntercept) != 0.0 or eval(self.yIntercept) != 0.0:
                plt.annotate(f"(0,{round(eval(self.yIntercept),3)})", # this is the text
                            (0,eval(self.yIntercept)), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(xText,yText), # distance from text to points (x,y)
                            ha='center',
                            fontproperties= font,
                            arrowprops = dict(arrowstyle="fancy",color=plotColor)) # horizontal alignment can be left, right or center
                plt.annotate(f"({round(eval(self.xIntercept),3)},0)", # this is the text
                            (eval(self.xIntercept),0), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(xText,yText), # distance from text to points (x,y)
                            ha='center',
                            fontproperties= font,
                            arrowprops = dict(arrowstyle="fancy",color=plotColor)) # horizontal alignment can be left, right or center
        elif self.a == 0:
            maximum = max(abs(eval(self.yIntercept)),
            max(tuple(map(lambda point: abs(point.x),self.SpecialPoints))) if len(self.SpecialPoints) > 0 else 0)
            x = np.linspace(-maximum-5,maximum+5,100)
            plt.plot(x,0*x,'r',linewidth=1) # label='x-axis',
            plt.plot(0*x,x,'r',linewidth=1) # label='y-axis',
            plt.plot(x,eval(self.yIntercept)+ 0*x,plotColor,
                        label=
                        (self.specialText + "\n" if self.specialText is not None else "" ) +
                        self.slopeInterceptFormat + '\n' +
                        self.interseptFormat + '\n' +
                        'Y-intercept: ' + self.yIntercept
                        ,linewidth=1)
            plt.scatter(0,eval(self.yIntercept),s=100,marker='o',cmap="Dark2_r",c=[eval(self.yIntercept)])
            plt.annotate(f"(0,{round(eval(self.yIntercept),.3)})", # this is the text
                        (0,eval(self.yIntercept)), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(xText,yText), # distance from text to points (x,y)
                        ha='center',
                        fontproperties= font,
                        arrowprops = dict(arrowstyle="fancy",color=plotColor)) # horizontal alignment can be left, right or center
        elif self.b == 0:
            maximum = max(abs(eval(self.xIntercept)),
            max(tuple(map(lambda point: abs(point.x),self.SpecialPoints))) if len(self.SpecialPoints) > 0 else 0)
            x = np.linspace(-maximum-5,maximum+5,100)
            plt.plot(x,0*x,'r',linewidth=1) # label='x-axis',
            plt.plot(0*x,x,'r',linewidth=1) # label='y-axis',
            plt.plot(0*x + eval(self.xIntercept),x,plotColor,
                    label= 
                    (self.specialText + "\n" if self.specialText is not None else "" ) +
                    self.__str__() + "\n"
                    # self.slopeInterceptFormat + '\n' +
                    # self.interseptFormat + '\n' +
                    'X-intercept: ' + self.xIntercept  + "\n"
                    ,linewidth=1)
            plt.scatter(eval(self.xIntercept),0,s=100,marker='o',cmap="Dark2_r",c=[0])
            plt.annotate(f"({round(eval(self.xIntercept),3)},0)", # this is the text
                        (eval(self.xIntercept),0), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(xText,yText), # distance from text to points (x,y)
                        ha='center',
                        fontproperties= font,
                        arrowprops = dict(arrowstyle="fancy",color=plotColor)) # horizontal alignment can be left, right or center

            
        
    @classmethod
    def linePairPlots(cls,style, *lines,externalPoints=[],isPlotAngle=False):
        cls.getReadyPlotObject(style)
        font = cls.font
        if len(lines) <=2 :
            cls.plotObject.title(f'Graph of:\n {"  and  ".join(map(str,lines))}',fontsize=14, fontweight='bold')
        else:
            cls.plotObject.title(f'Graph of:\n {"  and  ".join(map(str,lines[0:2]))} and \n {" and ".join(map(str,lines[2:]))}',
                            fontproperties=font)
        colorCodes = ['#6C3483','#FA64BF','#9C0821','#F39C12']
        tupleXYText = ((-30,25),(30,25),(30,-25),(-30,-25))
        max_distance = 0.0
        
        PointsList = []
        if len(lines) <= 4 :
            for line_1,line_2 in combinations(lines,2):
                pt = line_1.intersection(line_2)
                if pt not in PointsList:
                    cls.plotIntersection(pt)
                PointsList.append(pt)
                distance =  0 if pt is None else abs(pt.x)
                if max_distance < distance :
                    max_distance = distance
                    
            for i in range(len(lines)):
                lines[i].linePlot(plotObject=cls.plotObject,xyText=tupleXYText[i],plotColor=colorCodes[i],max_distance=max_distance)      
            
        else:
            raise ValueError("Can't plot more than 4 lines")
        
        # PointsList.extend(externalPoints)
        for pt in externalPoints:
            if pt not in PointsList:
                StraightLine.plotObject.scatter(pt.x,pt.y,s=100,marker='x',cmap="Dark2_r",c=[pt.y])
                StraightLine.plotObject.annotate(f"({round(pt.x,2)},{round(pt.y,2)})", # this is the text
                        (pt.x,pt.y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(-35, -40), # distance from text to points (x,y)
                        ha='center',
                        fontproperties= font,
                        arrowprops = dict(arrowstyle="simple",color="magenta")) # horizontal alignment can be left, right or center
                PointsList.append(pt)
                max_distance = max(max_distance,pt.x)
                # StraightLine.plotObject.show()
        
        if isPlotAngle:
            grad1 = math.degrees(math.atan(-lines[0].a / lines[0].b))
            grad2 = math.degrees(math.atan(-lines[1].a / lines[1].b))

            s_deg = min(grad1,grad2)
            e_deg = max(grad1,grad2)

            pt_of_intersection = lines[0].intersection(lines[1])

            if e_deg - s_deg < 30 :
                m = 20
            elif  e_deg - s_deg < 60:
                m = 15
            else:
                m= 3

            start_pt = Point(pt_of_intersection.x + m*math.cos(s_deg * math.pi / 180),
                            pt_of_intersection.y + m*math.sin(s_deg * math.pi / 180))

            end_pt = Point(pt_of_intersection.x + m*math.cos(e_deg * math.pi / 180),
                            pt_of_intersection.y + m*math.sin(e_deg * math.pi / 180))
            plt.plot((start_pt.x + end_pt.x+pt_of_intersection.x)/3, (start_pt.y +  end_pt.y + pt_of_intersection.y)/3, "",
                label=f"Angle between is \n{lines[0].acuteAngleBetween(lines[1])}")
            # plt.annotate("",
            #     xy=(start_pt.x, start_pt.y), xycoords='data',
            #     xytext=( end_pt.x, end_pt.y), textcoords='data',
            #     arrowprops=dict(arrowstyle="->", color="0.2",
            #                     shrinkA=5, shrinkB=5,
            #                     patchA=None, patchB=None,
            #                     connectionstyle="angle3,angleA=90,angleB=0",
            #                     ))

            # plt.annotate(f"{lines[0].acuteAngleBetween(lines[1])}",
            #     xy=(start_pt.x + 5, start_pt.y + 5), xycoords='data',
            #     xytext=( (end_pt.x + start_pt.x)/2, (start_pt.y+end_pt.y)/2), textcoords='offset points',
            #     arrowprops=dict(arrowstyle="fancy", color="red",
            #                     shrinkA=5, shrinkB=5,
            #                     patchA=None, patchB=None,
            #                     connectionstyle="angle3,angleA=90,angleB=0",
            #                     ))

            # plt.text(end_pt.x , end_pt.y,,fontsize=12) 

    

        cls.plotObject.legend()
        cls.plotObject.tight_layout()
        #cls.plotObject.grid(True)
        cls.plotObject.show()
    
    
    @classmethod
    def plotIntersection(cls,pt:Point=None):
        font = cls.font
        # font.set_family('monospace')
        #font.set_name('Times New Roman')
        #font.set_style('italic')
        font.set_weight('heavy')
        font.set_size('large')
        if pt is not None:
            cls.plotObject.scatter(pt.x,pt.y,s=100,marker='x',cmap="Dark2_r",c=[pt.y])
            cls.plotObject.annotate(f"({round(pt.x,2)},{round(pt.y,2)})", # this is the text
                    (pt.x,pt.y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, -40), # distance from text to points (x,y)
                    ha='center',
                    fontproperties= font,
                    arrowprops = dict(arrowstyle="fancy",color="red")) # horizontal alignment can be left, right or center
            
            
    @staticmethod
    def gcd(a, b):
        while b:
            a , b = b, a%b
        return a
    
    @staticmethod
    def lcm(a, b):
        return (a*b)//StraightLine.gcd(a,b)    
    
    @staticmethod
    def lcmExtended(args):
        return reduce(lambda x,y: StraightLine.lcm(x,y),args)
    
    @staticmethod
    def gcdExtended(args):
        return reduce(lambda x,y: StraightLine.gcd(x,y),args)
    
    @staticmethod
    def gcdSpecial(*args):
        args = tuple(map(lambda x: Fraction(f"{x}").limit_denominator(100), args))
        gcd_numerators = StraightLine.gcdExtended([i.numerator for i in args])
        lcm_denominators = StraightLine.lcmExtended([i.denominator for i in args])
        gcd_fraction =gcd_numerators / lcm_denominators# Fraction(gcd_numerators, lcm_denominators)
        args = tuple(map(lambda x: x.numerator / (gcd_fraction * x.denominator), args))
        return args
    
    @staticmethod
    def is_square(i):
        return i == math.isqrt(i) ** 2
    
    @staticmethod
    def isSameSign(a,b):
        if (a <= 0 and b <= 0) or (a >= 0 and b >= 0):
            return True
        else:
            return False
    
    def valueAt(self,point: Point = Point(0,0)):
        return self.a * point.x + self.b * point.y + self.c
    
    def perpendicularBisector(self,point1,point2):
        if self.valueAt(point1) != 0:
            print(f"{point1} is not on the line")
            return None
        if self.valueAt(point2) != 0:
            print(f"{point2} is not on the line")
            return None
        
        mp = Point.midPoint(point1,point2)
        return StraightLine(point1=mp, point2=mp, slope= (-1/self.exactSlope),specialText="Perpendicular Bisector")
    
    def angleBisector(self,line,which="acute"):
        grad1 = math.degrees(math.atan(-self.a / self.b))
        grad2 = math.degrees(math.atan(-line.a / line.b))

        s_deg = min(grad1,grad2)
        e_deg = max(grad1,grad2)
        pt_of_intersection = self.intersection(line)
        
        if (e_deg - s_deg) <= 90:
            bisec_grad = s_deg + ((e_deg - s_deg)/2)
        else:
            bisec_grad = s_deg + ((e_deg - s_deg)/2) - 90.0

        if which == "acute":
            bisec_slope = math.tan(bisec_grad*math.pi/180)
            return StraightLine(point1=pt_of_intersection,slope=bisec_slope,specialText="Acute Angle Bisector")
        if which == "obtused":
            bisec_slope = math.tan(bisec_grad*math.pi/180 + (math.pi / 2))
            return StraightLine(point1=pt_of_intersection,slope=bisec_slope,specialText="Obtused Angle Bisector")
        # if which == "acute":
        #     if (e_deg - s_deg) <= 90:
        #         bisec_grad = s_deg + ((e_deg - s_deg)/2)
        #     else:
        #         bisec_grad = s_deg + ((e_deg - s_deg)/2) - 90.0

        #     bisec_slope = math.tan(bisec_grad*math.pi/180)
        #     return StraightLine(point1=pt_of_intersection,slope=bisec_slope,specialText="Acute Angle Bisector")
        # if which == "obtused":
        #     if e_deg - s_deg <= 90:
        #         bisec_grad = s_deg + ((e_deg - s_deg)/2) + 90
        #     else:
        #         bisec_grad = s_deg + ((e_deg - s_deg)/2) 

        #     bisec_slope = math.tan(bisec_grad)
        #     return StraightLine(point1=pt_of_intersection,slope=bisec_slope,specialText="Obtused Angle Bisector")
            



    def relativePosition(self, point1: Point = Point(0,0), point2: Point = Point(0,0)):
        if point1 == Point(0,0) and point2 == Point(0,0):
            raise ValueError("Relative Position can not be found")
        else:
            if StraightLine.isSameSign(self.valueAt(point1),self.valueAt(point2)):
                return f"{repr(point1)} and {repr(point2)} are at Same Side of the line {self}"
            else:
                return f"{repr(point1)} and {repr(point2)} are at Opposite Side of the line {self}"
            
    def imageOf(self,point: Point = Point(0,0)):
        equatingFactor = ((-2) *(self.valueAt(point)))/(self.a * self.a + self.b * self.b)
        x = point.x + (self.a * equatingFactor)        
        y = point.y + (self.b * equatingFactor)    
        return Point(x, y)
    
    def footOfPerpendicularFrom(self,point: Point = Point(0,0)):
        equatingFactor = ((-1) *(self.valueAt(point)))/(self.a * self.a + self.b * self.b)
        x = point.x + (self.a * equatingFactor)        
        y = point.y + (self.b * equatingFactor)    
        return Point(x, y)

    def distanceFrom(self,point: Point = None,line=None):
        denom = (self.a * self.a + self.b * self.b)**0.5
        if point is not None:
            return abs(self.valueAt(point))/denom
        else:
            if self.isParallel(line):
                return abs(self.c - line.c)/denom
            else:
                # return "0.0, Intersecting Lines"
                return 0.0
            
    def intersection(self,other):
        try:
            x = (self.b * other.c - self.c * other.b) / (self.a * other.b - self.b * other.a)
            y = (self.c * other.a - self.a * other.c) / (self.a * other.b - self.b * other.a)
            return Point(x,y)
        except ZeroDivisionError:
            # print("No point of intersection found, Those are parallel lines")
            return None
    
    @property
    def coefX(self):
        return self.a
    
    @property
    def coefY(self):
        return self.b
    
    @property
    def slope(self):
        if self.b != 0:
            return f"({round(-self.a / self.b,3)}) or tan({round(math.degrees(math.atan(-self.a / self.b)),3)})"
        else:
            return f"Undefined Slope or tan(90.0)"
    
    @property
    def exactSlope(self):
        return -self.a / self.b if self.a != 0 else math.nan
    @property
    def pointSlopeFormat(self):
        idx = self.slope.index('or')
        return  f"(y - {round(self.SpecialPoints[0].ordinate,2)}) = {self.slope[idx+2:]} * (x - {round(self.SpecialPoints[0].abscissa,2)})"
        
    @property
    def twoPointFormat(self):
        return f"line joining {self.SpecialPoints[0]} and {self.SpecialPoints[1]}"
        
    @property
    def normalFormat(self):
        A = self.a
        B = self.b
        C = self.c
        if C >= 0:
            A = -A
            B = -B
        else:
            C = -C
        
        if A > 0 and B >= 0:
            degree = round(math.degrees(math.atan(abs(B/A))),2) 
        elif A < 0 and B >= 0:
            degree = round(180 - math.degrees(math.atan(abs(B/A))),2) 
        elif A < 0 and B <= 0:
            degree = round(180 + math.degrees(math.atan(abs(B/A))),2) 
        elif A > 0 and B <= 0:
            degree = round(360 - math.degrees(math.atan(abs(B/A))),2) 
        elif A == 0:
            if B > 0:
                degree = round(90,2)
            else:
                degree = round(270,2)
        P = round(C / (A*A + B*B)**0.5,2)
        
        return f"x cos({degree}) + y sin({degree}) = {P}"
    

    @property
    def generalFormat(self):
        if self.a != 0 and self.b != 0 and self.c != 0:
            self.a, self.b, self.c = StraightLine.gcdSpecial(self.a, self.b, self.c)
            return f"({round(self.a,3)}) x + ({round(self.b,3)}) y + ({round(self.c,3)}) = 0"
        elif self.a == 0 and self.b != 0 and self.c != 0:
            return f"({round(self.b,3)}) y + ({round(self.c,3)}) = 0"
        elif self.a != 0 and self.b == 0 and self.c != 0:
            return f"({round(self.a,3)}) x + ({round(self.c,3)}) = 0"
        elif self.a != 0 and self.b != 0 and self.c == 0:
            return f"({round(self.a,3)}) x + ({round(self.b,3)}) y = 0"
        elif self.b == 0 and self.c == 0:
            return "x = 0"
        elif self.a == 0 and self.c == 0:
            return "y = 0"


    @property
    def slopeInterceptFormat(self):
        if self.b != 0:
            if any((type(self.a) is float, type(self.b) is float, type(self.c) is float)):
                return f"y = ({round(-self.a / self.b,3)}) x + ({round(-self.c / self.b,3)})"
            else:
                return f"y = ({Fraction(f'{-self.a/self.b}').limit_denominator(100)}) x + ({Fraction(f'{-self.c / self.b}').limit_denominator(100)})"
        else:
            return f"x = ({Fraction(f'{-self.c/self.a}').limit_denominator(100)})" #""""Undefined Slope" + "\n"+"""
    
    @property
    def interseptFormat(self):
        if self.a == 0:
            if any((type(self.b) is float, type(self.c) is float)):
                return  f"y = {round(-self.c/self.b,3)}"
            else:    
                return f"y = {Fraction(f'{-self.c/self.b}').limit_denominator(1000)}"
        elif self.b == 0:
            if any((type(self.a) is float, type(self.c) is float)):
                return  f"y = {round(-self.c/self.a,3)}"
            else:             
                return f"x = {Fraction(f'{-self.c/self.a}').limit_denominator(100)}"
        elif self.c == 0:
            return "Passing through Origin"
        else:
            if any((type(self.a) is float, type(self.b) is float, type(self.c) is float)):
                return f"x / ({round(-self.c/self.a,3)}) + y / ({round(-self.c/self.b,3)}) = 1"
            else:
                return f"x / ({Fraction(f'{-self.c/self.a}').limit_denominator(100)}) + y / ({Fraction(f'{-self.c/self.b}').limit_denominator(100)}) = 1"
    
    @property
    def xIntercept(self):
        if self.a != 0:
            if any((type(self.a) is float, type(self.c) is float)):
                return  f"({round(-self.c/self.a,3)})"
            else: 
                return f"{Fraction(f'{-self.c/self.a}').limit_denominator(100)}"
        else:
            return None # "This line is parallel to X-axis"

    @property
    def yIntercept(self):
        if self.b != 0:
            if any((type(self.b) is float, type(self.c) is float)):
                return  f"({round(-self.c/self.b,3)})"
            else: 
                return f"{Fraction(f'{-self.c/self.b}').limit_denominator(100)}"
        else:
            return None # "This line is parallel to Y-axis"
    
    @property
    def constant(self):
        return self.c
    
    def isParallel(self,other):
        try:
            if ((self.a / other.a) == (self.b / other.b)) and ((self.b / other.b) != (self.c / other.c)):
                return True
            else:
                return False
        except ZeroDivisionError:
            if self.intersection(other) is None:
                return True
            else:
                return False
            
    def isPerpendicular(self, other):
        try:    
            frac1 = Fraction(f"{-self.a/self.b}") 
            frac2 = Fraction(f"{-other.a/other.b}")
            if frac1 * frac2.numerator == -1:
                return True
            else:
                return False
        except ZeroDivisionError:
            if self.intersection(other) is None:
                return False
            else:
                return True
    
    def isIntersecting(self, other):
        if (self.a / other.a) != (self.b / other.b):
            return True
        else:
            return False
    
    def isCoincident(self, other):
        if (self.a / other.a) == (self.b / other.b) == (self.c / other.c):
            return True
        else:
            return False
            
    def acuteAngleBetween(self,other):
        try:
            value = abs((self.a * other.b - self.b * other.a) / (self.a * other.a + self.b * other.b)) 
            deg = math.degrees(math.atan(abs(value)))
            return f"{round(deg,2)} degrees or PI*({Fraction(f'{math.radians(deg) / math.pi}').limit_denominator(100)}) radians"
        except ZeroDivisionError:
            #({Fraction(f'{math.radians(90) / math.pi}')})
            return f"90.0 degrees or PI/2 radians"
        
    def __repr__(self):
        return f"StraightLine({self.a},{self.b},{self.c})"
    
    def __str__(self):
        if self.isInterseptForm:
            return self.interseptFormat
        elif self.isNormalForm:
            return self.normalFormat
        elif self.isPointSlopeForm:
            return self.pointSlopeFormat
        elif self.isTwoPointForm:
            return self.twoPointFormat
        elif self.isSlopeInterceptForm:
            return self.slopeInterceptFormat
        else:
            return self.generalFormat
        

def configure_line(lines):
        while True:
            print("\n---------------------Pick a Format: ------------------------\n")
            print("1. General Format    :=>  ax + by + c = 0 \n"+
                "\n  ==>Description "+
                "\n    --------------"+
                "\n      This is also well known as linear equation in x and y "
                "\n      Taking the values of the co-efficients of x and y and the constant term"+
                "\n      i.e. taking the values of a, b and c\n")
            print("2. Point Slope Format    :=>  (y - y\N{SUBSCRIPT ONE}) = m (x - x\N{SUBSCRIPT ONE})" +
                "\n\n  ==>Description "+
                "\n    --------------"+
                "\n      Taking a point (x1,y1) through which the line should pass and the slope(m or tan \u03B8 ) of the line"+
                "\n      where \u03B8 being the angle made by the line with positive direction of x-axis\n")
            print("3. Two Point Format     :=>   (y - y\N{SUBSCRIPT ONE}) / (x - x\N{SUBSCRIPT ONE}) = (y\N{SUBSCRIPT TWO} - y\N{SUBSCRIPT ONE}) / (x\N{SUBSCRIPT TWO} - x\N{SUBSCRIPT ONE}) "+
                "\n\n  ==>Description "+
                "\n    --------------"+
                "\n      Taking two points (x\N{SUBSCRIPT ONE},y\N{SUBSCRIPT ONE}) and (x\N{SUBSCRIPT TWO},y\N{SUBSCRIPT TWO}) respectively through which the line should pass\n")
            print("4. Intercept Format    :=>   (x / a) + (y / b) = 1"+
                "\n\n  ==>Description "+
                "\n    --------------"+
                "\n      Taking the value of x-Intercept(a) and y-Intercept(b)\n")
            print("5. Normal Format     :=>   x cos(\u03B1) + y sin(\u03B1) = p"+
                "\n\n  ==>Description "+
                "\n    --------------"+
                "\n      Taking p as the distance of the line from origin and "+
                "\n       \u03B1 as the angle between positive x-axis and the perpendicular line through the origin\n")
            print("6. Slope Intercept Format    :=>   y = mx + c"+
                "\n\n  ==>Description "+
                "\n    --------------"+
                "\n      Taking the slope(m or tan \u03B8 )  and y-Intercept(c) of the line"+
                "\n      where \u03B8 being the angle made by the line with positive direction of x-axis\n")
            print("7. Exit from this section\n")
            option = eval(input("Choose an option from the above list: "))
            if option == 1:
                A,B,C = map(eval,input("Enter values of a, b and c separated by spaces: ").split())
                lines.append(StraightLine(a=A, b=B, c=C))
            elif option == 2:
                x, y = map(eval,input("Enter coordinates of the point separated by spaces: ").split())
                slope = eval(input("Enter slope of the line: "))
                lines.append(StraightLine(point1=Point(x,y),slope=slope))
            elif option == 3:
                x1, y1 = map(eval,input("Enter coordinates of the First Point separated by spaces: ").split())
                x2, y2 = map(eval,input("Enter coordinates of the Second Point separated by spaces: ").split())
                lines.append(StraightLine(point1=Point(x1,y1),point2=Point(x2,y2)))
            elif option == 4:
                A,B = map(eval,input("Enter values of x-Intercept and y-Intercept separated by spaces: ").split())
                lines.append(StraightLine(xIntercept=A,yIntercept=B))
            elif option == 5:
                p = eval(input("Enter the distance of the line from the origin: "))
                alpha = eval(input("Enter the value of alpha: "))
                lines.append(StraightLine(p=p,alpha=alpha))
            elif option == 6:
                slope = eval(input("Enter slope of the line: "))
                B = eval(input("Enter y-Intercept of the line: "))
                lines.append(StraightLine(slope=slope,yIntercept=B))
            else:
                break
            
            accept_any_key()

def print_lines(lines):
    print("\n----------List of lines your have specified -----------\n")
    s="\n----------List of lines your have specified -----------\n\n"
    for i,line in enumerate(lines):
        print(f"Line No {i+1}.  {line}")
        s+=f"Line No {i+1}.  {line}\n"
    print("\n---------------------------------------------------------\n")
    s+="\n---------------------------------------------------------\n"
    
    print("** N.B. : Please make a note of the line numbers correspoding to a particular line.")
    print("You will need this to refer this line number whenever you need to access that particular straight line**")
    print("                                       or, ")
    print("You can save it to file for future refrence...........")
    
    if "y" in input("Do you want to store this details to a file for future refrence?[y/n] : ").lower():
        save_path = '.\Saved Report Files'
        file_name = input("\nEnter name of the file: ")
        file_name = file_name+".txt"
        completeName = path.join(save_path, file_name)
        with open(completeName, 'w') as f:
            f.write(s)
        print(f"Details successfully saved in the file {file_name}")
        print("You can find the same in your current directory........")
        
        
    accept_any_key()


def plot_specified_lines(lines):
    ip = input("\nEnter the line numbers you want to plot(separated by spaces) \n" + \
    "e.g. Suppose you want to plot lines 1 ,3 and 4 then just specify 1,3,4 \n" + \
    "Specify your choices: ")
    requiredLines = [lines[i-1] for i in map(eval,ip.split(','))]
    StraightLine.linePairPlots(style,*requiredLines)
    accept_any_key()

def display_details(line):
    print("\n********************************************************")
    
    print("=> Basic Properties  ------------------------------------\n")
    print(f"    Slope                   : {line.slope}")
    print(f"    X-Intercept             : {line.xIntercept}")
    print(f"    y-Intercept             : {line.yIntercept}")
    print(f"    Cuts the x-axis at      : {None if line.xIntercept is None else Point(line.xIntercept,0)}")
    print(f"    Cuts the y-axis at      : {None if line.yIntercept is None else Point(0,line.yIntercept)}")
    print(f"    Distance from origin    : {round(line.distanceFrom(point=Point(0,0)),3)}\n")
    
    print("=> Various formats   -------------------------------------\n")
    print(f"    General Form            : {line.generalFormat}")
    print(f"    Slope Intercept Form    : {line.slopeInterceptFormat}")
    print(f"    Intercept Form          : {line.interseptFormat}")
    print(f"    Normal Form             : {line.normalFormat}")
    if line.isTwoPointForm:
        print(f"    Two Point Form          : {line.twoPointFormat}")
    else:
        print(f"    Two Point Form          : Not Applicable")
    if line.isPointSlopeForm:
        print(f"    Point Slope Form        : {line.pointSlopeFormat}")
    else:
        print(f"    Point Slope Form        : Not Applicable")
        
    print("\n*******************************************************\n")
    
    s=""
    s+="\n********************************************************\n"
    s+="\nDetails about the line your have specified is .......\n\n"
    s+="=> Basic Properties  ------------------------------------\n\n"
    s+=f"    Slope                   : {line.slope}\n"
    s+=f"    X-Intercept             : {line.xIntercept}\n"
    s+=f"    y-Intercept             : {line.yIntercept}\n"
    s+=f"    Cuts the x-axis at      : {None if line.xIntercept is None else Point(line.xIntercept,0)}\n"
    s+=f"    Cuts the y-axis at      : {None if line.yIntercept is None else Point(0,line.yIntercept)}\n"
    s+=f"    Distance from origin    : {round(line.distanceFrom(point=Point(0,0)),3)}\n\n"
    s+="=> Various formats   -------------------------------------\n\n"
    s+=f"    General Form            : {line.generalFormat}\n"
    s+=f"    Slope Intercept Form    : {line.slopeInterceptFormat}\n"
    s+=f"    Intercept Form          : {line.interseptFormat}\n"
    s+=f"    Normal Form             : {line.normalFormat}\n"
    if line.isTwoPointForm:
        s+=f"    Two Point Form          : {line.twoPointFormat}\n"
    else:
        s+=f"    Two Point Form          : Not Applicable\n"
    if line.isPointSlopeForm:
        s+=f"    Point Slope Form        : {line.pointSlopeFormat}\n"
    else:
        s+=f"    Point Slope Form        : Not Applicable\n"
        
    s+="\n*******************************************************\n\n"
    if "y" in input("Do you want to store this details to a file for future refrence?[y/n] : ").lower():
        save_path = '.\Saved Report Files'
        file_name = input("\nEnter name of the file: ")
        file_name = file_name+".txt"
        completeName = path.join(save_path, file_name)
        with open(completeName, 'w') as f:
            f.write(s)
        print(f"Details successfully saved in the file {file_name}")
        print("You can find the same in your current directory........")
    accept_any_key()

def single_line_properties(style,line):
    print("=========================================================")
    print("  Details about the line your have specified is .......")
    print("=========================================================")
    display_details(line)
    print("=========================================================")
    print("********************************************************************")
    print("**  Explore Various Properties related to a Single Straightlines  **")
    print("********************************************************************")
    print(r"                       |||")
    print(r"                       |||")
    print(r"                      \   /")
    print(r"                       \ /")
    
    while True:
        print("\n======================== Choose a topic from the list given below ==========================\n")
        print("1. Foot of the perpendicular drawn from a given point to the line and the distance between those")
        print("2. Image of a given point with respect to the line")
        print("3. Relative position of two given points with respect to the line")
        print("4. Perpendicular Bisector of a portion of the line between the two given points")
        print("5. Exit from this section\n")
        option = eval(input("Choose an option from the above list: "))
        print("------------------------------------------------------------------------------------------\n")
        if option == 1:
            x, y = map(eval,input("Enter coordinates of the point separated by spaces: ").split())
            pt = Point(x,y)
            Foot = line.footOfPerpendicularFrom(point=pt)
            print(f"Foot of the perpendicular drawn from the point {pt} to the line {line} is: {Foot},")
            print(f"and the distance of {pt} from the given line {line} is {round(line.distanceFrom(point=pt),3)}")
            p_slope = -1/line.exactSlope
            print(f"The equation of the perpendicular line passing through {pt} is {StraightLine(point1=Foot,slope=p_slope).slopeInterceptFormat}")
            StraightLine.linePairPlots(style,line,StraightLine(point1=pt,point2=Foot))
        elif option == 2:
            x, y = map(eval,input("Enter coordinates of the point separated by spaces: ").split())
            pt = Point(x,y)
            Image = line.imageOf(point=pt)
            print(f"Image of the point {pt} with respect to the line is: {Image}")
            p_slope = -1/line.exactSlope
            StraightLine.linePairPlots(style,line,StraightLine(point1=Image,point2=pt))
        elif option == 3:
            x1, y1 = map(eval,input("Enter coordinates of the First point separated by spaces: ").split())
            pt1 = Point(x1,y1)
            x2, y2 = map(eval,input("Enter coordinates of the Second point separated by spaces: ").split())
            pt2 = Point(x2,y2)
            print(line.relativePosition(point1=pt1,point2=pt2))
            StraightLine.linePairPlots(style,line,externalPoints=[pt1,pt2])
        elif option == 4:
            x1, y1 = map(eval,input("Enter coordinates of the First point separated by spaces: ").split())
            pt1 = Point(x1,y1)
            x2, y2 = map(eval,input("Enter coordinates of the Second point separated by spaces: ").split())
            pt2 = Point(x2,y2)
            pb = line.perpendicularBisector(point1=pt1,point2=pt2)
            if pb is not None:
                print(f"Perpendicular Bisector of the line between {pt1} and {pt2} is: {pb}")
                StraightLine.linePairPlots(style,line,pb,externalPoints=[pt1,pt2]) 
            else:
                print(f"Not possible to find the perpendicular bisector of the line between {pt1} and {pt2}")
                print(f"as they don't lie on the same line {line}")
                StraightLine.linePairPlots(style,line,externalPoints=[pt1,pt2]) 
        else:
            break
        print("------------------------------------------------------------------------------------------\n")
        accept_any_key()

def pair_line_properties(style,line1=None, line2=None):
    print("===============================================================")
    print("  The details about the Line 1 you have specified is.........")
    display_details(line1)
    print("===============================================================")
    print("\nThe details about the Line 2 you have specified is.........")
    display_details(line2)
    print("===============================================================")
    print()
    print("***********************************************************************")
    print("**  Explore Various Properties related to the pair of straightlines  **")
    print("***********************************************************************")
    print(r"                                |||")
    print(r"                                |||")
    print(r"                               \   /")
    print(r"                                \ /")
    
    if line1.isParallel(line2):
        print("\n=> The given lines are parallel to each other")
        print(f"and the distance between these two lines is : {line1.distanceFrom(line=line2)}")
    else:
        print("\n=> The given lines are intersecting")
        print(f"The point of intersection is {line1.intersection(line2)}")
        print(f"\n=> Acute angle between two given straight lines is : {line1.acuteAngleBetween(line2)}")
        if line1.isPerpendicular(line2):
            print("\n=> The given two given straight lines are perpendicular to each other")
        else:
            print("\nThe given two given straight lines are not perpendicular to each other")
            
        ab_acute = line1.angleBisector(line2)
        ab_obtused = line1.angleBisector(line2,which='obtused')
        print(f"\n=> The equation of the acute angle bisector is : {ab_acute}")
        print(f"\n=> The equation of the obtused angle bisector is : {ab_obtused}")
    if "y" in input("\nDo you want to plot angle bisector?[y/n] : "):
        print("1. For Acute angle Bisector")
        print("2. For Obtused Angle Bisector")
        ch=eval(input("Enter your choice: "))
        if ch == 1:
            StraightLine.linePairPlots(style,line1,line2,ab_acute,isPlotAngle=True)
        elif ch == 2:
            StraightLine.linePairPlots(style,line1,line2,ab_obtused,isPlotAngle=True)
        else:
            print("Choose your option correctly and try again......")
    print("-------------------------------------------------------------------------------------\n")
    accept_any_key()

if __name__ == "__main__":
    mkdir('Saved Report Files') 
    lines = []
    st1 = StraightLine(a=2,b=3,c=8)
    st2 = StraightLine(point1=Point(2,8),slope=5)
    st3 = StraightLine(point1=Point(7,8),point2=Point(-9,2))
    st4 = StraightLine(xIntercept=4,yIntercept=7)
    st5 = StraightLine(alpha = 135,p = 10)
    st6=StraightLine(slope=8,yIntercept=10)
    lines=[st1,st2,st3,st4,st5,st6]

    set_font_style()

    print("=============================================================================")
    print("///////////     Welcome to Vsual Analysis of a well known         ///////////")
    print("///////////   two dimensional geometric figure *Straightlines*    ///////////")
    print("============================================================================\n")
    
    print("****************************************************************************")
    print("**     The initial step to follow when you first start this software      **")
    print("**   ------------------------------------------------------------------   **")
    print("**      First of all configure one or few straightlines by choosing       **")
    print("**     the option 1 from the list given below and then proceed further    **")
    print("**   ------------------------------------------------------------------   **")
    print("**       You can find the list of straightlines you have specified        **")
    print("**    in option 2 as and when you need, also you can save it to a file.   **")
    print("**    You can find all the files in the directory '/Saved Report Files'   **")
    print("**           in you installation directory of the software                **")
    print("**   ==================================================================   **")
    print("**      N.B. : Whenever you need to enter the coordinates of a point      **")
    print("**    then you just only need to mentioned the coordinates separated by   **")
    print("**      spaces i.e. to enter the point (4,7) you should write => 4 7      **")
    print("**        or, to enter the point (-3,8) you should write  => -3 8         **")
    print("**        or, to enter the point (-2,-9) you should write => -2 -9        **")
    print("****************************************************************************")
    print(r"                                  |||")
    print(r"                                  |||")
    print(r"                                 \   /")
    print(r"                                  \ /")
    while True:   
        try:
            print("\n--------------------- Choose a Topic: ------------------------\n")
            print("1. Configure Straight Lines in various format")
            print("2. Display the list of all configured lines")
            print("3. Display the details of a specified straight line")
            print("4. Anlyzing properties related to a single line with visual description")
            print("5. Anlyzing properties related to pair of straight lines with visual description")
            print("6. Plot one or more strightlines [supports maximum 4 simultaneous plots] ")
            print("7. Exit from this software\n")
            option = eval(input("Choose an option from the above list: "))
            
            if 1< option < 7 and len(lines)==0:
                print("Kindly configure one or few straightlines to proceed further......")
                print("You can do so by choosing the option 1 from the list")
                accept_any_key()
                continue
            
            if option == 1: 
                configure_line(lines)
            elif option == 2:
                print_lines(lines)
            elif option == 3:
                if len(lines) == 0:
                    continue
                choice = eval(input("Enter the line number you want to see the details: "))
                display_details(lines[choice-1])
            elif option == 4:
                choice = eval(input("Enter the line number you want to explore its properties: "))
                single_line_properties(style,lines[choice-1])
            elif option == 5:
                ch1,ch2 = map(eval,input("Specify the line numbers of the pair of lines separated by spaces: ").split())
                pair_line_properties(style,line1=lines[ch1-1],line2=lines[ch2-1])
            elif option == 6:
                plot_specified_lines(lines)
            else:
                print("\nExiting from the system..........")
                accept_any_key()
                break
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("The system has encountered an unexpected problem!!! Check the input values you are giving and try again !!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Description of the error: {e}")




# st1 = StraightLine(a=2,b=3,c=8)
# st2 = StraightLine(point1=Point(2,8),slope=5)
# st3 = StraightLine(point1=Point(7,8),point2=Point(-9,2))
# st4 = StraightLine(xIntercept=4,yIntercept=7)
# st5 = StraightLine(alpha = 135,p = 10)
# st6=StraightLine(slope=8,yIntercept=10)
# L=[st1,st2,st3,st4,st5,st6]
# for l1,l2 in it.combinations(L,2):
#     pair_line_properties(style,l1,l2)

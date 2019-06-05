# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:54:28 2018

@author: supersaiyajin7
"""
import plotly
plotly.tools.set_credentials_file(username='abc', api_key='xyz')
import plotly.plotly as ply
import plotly.graph_objs as go


def plot_3d(list_dict):
    z1 = list(list_dict.values())
    layout = go.Layout(
                        scene = dict(
                        xaxis = dict(
                            title='time in seconds'),
                        yaxis = dict(
                            ticktext=  ["Angry", "Disgust",
                         "Fear", "Happy",
                         "Sad", "Surprise",
                         "Neutral"],
                            tickvals= [0,1,2,3,4,5,6],
                            title='emotions'),
                        zaxis = dict(
                            title='value',
                            tickvals= [0,0.2,0.4,0.6,0.8],
                            ticktext = [0,20,40,60,80]
                            ),),
                        width=700,
                        margin=dict(
                        r=20, b=10,
                        l=10, t=10)
                      )
        
    data = [
        go.Surface(z=z1)
        #go.Surface(z=z2,showscale=False)    
    ]
    
    fig = go.Figure(data=data, layout=layout)
    #ply.plot(fig)#,filename='python-docs/multiple-surfaces')
    ply.image.save_as(fig, filename='C:/Users/shringar.kashyap/Desktop/mer_ind/mervyn_index/index/img/plot420.png')
    
    
    


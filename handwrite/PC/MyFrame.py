# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc


###########################################################################
## Class MyFrame
###########################################################################

class MyFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                          size=wx.Size(500, 290), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHintsSz(wx.Size(500, 290), wx.Size(500, 290))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"Handwriting Input", wx.DefaultPosition, wx.DefaultSize,
                                           wx.ALIGN_CENTRE)
        self.m_staticText2.Wrap(-1)
        self.m_staticText2.SetFont(wx.Font(20, 70, 90, 92, False, wx.EmptyString))
        self.m_staticText2.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_SCROLLBAR))

        bSizer2.Add(self.m_staticText2, 1, wx.ALL, 5)

        bSizer1.Add(bSizer2, 0, wx.EXPAND, 5)

        bSizer3 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_textCtrl = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(-1, 120),
                                      wx.TE_MULTILINE)
        self.m_textCtrl.SetFont(wx.Font(25, 70, 90, 92, False, wx.EmptyString))

        bSizer3.Add(self.m_textCtrl, 1, wx.ALL, 5)

        bSizer1.Add(bSizer3, 1, wx.EXPAND, 5)

        bSizer4 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_button_start = wx.Button(self, wx.ID_ANY, u"START", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button_start, 1, wx.ALL, 5)

        self.m_button_stop = wx.Button(self, wx.ID_ANY, u"CLEAR", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button_stop, 1, wx.ALL, 5)

        bSizer1.Add(bSizer4, 0, wx.EXPAND, 5)

        bSizer9 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText5 = wx.StaticText(self, wx.ID_ANY, u"IP", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText5.Wrap(-1)
        self.m_staticText5.SetFont(wx.Font(20, 70, 90, 90, False, wx.EmptyString))

        bSizer9.Add(self.m_staticText5, 0, wx.ALL, 5)

        self.m_staticText_ip = wx.StaticText(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_ip.Wrap(-1)
        self.m_staticText_ip.SetFont(wx.Font(20, 70, 90, 90, False, wx.EmptyString))

        bSizer9.Add(self.m_staticText_ip, 1, wx.ALL, 5)

        self.m_staticText_model = wx.StaticText(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_model.Wrap(-1)
        self.m_staticText_model.SetFont(wx.Font(20, 70, 90, 92, False, wx.EmptyString))
        self.m_staticText_model.SetForegroundColour(wx.Colour(0, 255, 128))
        self.m_staticText_model.SetMinSize(wx.Size(200, -1))
        self.m_staticText_model.SetMaxSize(wx.Size(200, -1))

        bSizer9.Add(self.m_staticText_model, 0, wx.ALL, 5)

        bSizer1.Add(bSizer9, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.m_button_start.Bind(wx.EVT_BUTTON, self.start)
        self.m_button_stop.Bind(wx.EVT_BUTTON, self.stop)

    def __del__(self):
        pass

    # Virtual event handlers, overide them in your derived class
    def start(self, event):
        event.Skip()

    def stop(self, event):
        event.Skip()



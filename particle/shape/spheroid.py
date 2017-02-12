# -*- coding: utf-8 -*-

from .shapeslice import shapeslice

class spheroid(shapeslice):
    '''
        楕円体クラス。
        内部変数としてx軸、y軸、z軸の半分の長さ、および中心をもつ。
        <remarks/>
            将来的にはEuler回転に対応させたい。
        </remarks>
    '''
    def __init__(self, ax, **kwargs):
        self._shape_name = 'spheroid'
        self.ax = ax
        self.ay = ax if kwargs.get('ay') is None else kwargs.get('ay')
        self.az = ax if kwargs.get('az') is None else kwargs.get('az')
        if self.ax == self. ay and self.ay == self.az:
            self._shape_name = 'sphere'
        self.center = [0., 0., 0.] if kwargs.get('center') is None else kwargs.get('center')

        self.a = max(self.ax, self.ay, self.az)
        self.a_range = self.a*1.1
        shapeslice.__init__(self, self._shape_name, self.ax, self.ay, self.az,
                            center=self.center)
        self._kwargs = kwargs

    def shape_name(self):
        return self._shape_name + ""

    def info(self):
        """
            particleshapeメソッドで生成するための情報を返す
            モデル生成に必要なものと、外部で利用する値を返す
        """
        if self._shape_name == 'sphere':
            _kwargs = dict(a_range=self.a_range, center=self.center)
        else:
            _kwargs = dict(a_range=self.a_range, center=self.center, ay=self.ay, az=self.az)
        return dict(shape_name=self._shape_name, a=self.ax, kwargs=_kwargs)

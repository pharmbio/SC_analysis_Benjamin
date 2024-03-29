from dataclasses import dataclass, astuple, asdict, replace
from typing import Union, List, Dict, Literal

@dataclass
class ClipSquare:
    '''
    A square crop centered at (x, y) with a given side.
    '''
    x: float
    y: float
    side: float

    def to_str(self):
        return ','.join(map(str, astuple(self)))

@dataclass
class ClipBox:
    '''
    A rectuangular crop with upper left corner (x, y) and sides (w, h).
    '''
    x: float
    y: float
    w: float
    h: float

    def to_str(self):
        return ','.join(map(str, astuple(self)))

@dataclass
class View:
    barcode: str = ''
    well: str = ''
    site: Union[int, None] = None
    clip: Union[ClipSquare, ClipBox, None] = None
    x: Union[int, None] = None
    y: Union[int, None] = None
    hover: Union[str, None] = None
    overlay: Union[str, None] = None
    overlay_dir: Literal['N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW', 'C', None] = None
    overlay_style: Union[str, None] = None

    def to_dict(self):
        out: Dict[str, Union[int, str]] = asdict(self)
        if self.clip:
            out['clip'] = self.clip.to_str()
        return {
            k: v
            for k, v in out.items()
            if v is not None
        }

    def to_str(self):
        from urllib.parse import quote_plus
        return '&'.join([
            f'{quote_plus(k)}={quote_plus(str(v))}'
            for k, v in self.to_dict().items()
        ])

@dataclass
class Viewer:
    '''
    v0.1
    '''
    views: Union[View, List[View]]
    width: Union[str, int]='100%'
    height: Union[str, int]='800px'

    @property
    def views_as_list(self):
        if not isinstance(self.views, list):
            return [self.views]
        else:
            return self.views

    def to_url(self):
        base = 'https://im.devserver.pharmb.io/v0.1.html'
        hash = '&'.join([view.to_str() for view in self.views_as_list])
        return f'{base}#{hash}'

    def to_html(self):
        src = self.to_url().replace('"', '&quot;')
        width = str(self.width).replace('"', '&quot;')
        height = str(self.height).replace('"', '&quot;')
        print(f'<iframe src="{src}" width="{width}" height="{height}"></iframe>') 
        return f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
       
    _repr_html_ = to_html
    

example1 = Viewer(
    [
        View(
            barcode='P101337',
            well=f'{row}{col}',
            site=site_x + site_y * 3 - 3,
            x=well_x * 3 + site_x,
            y=well_y * 3 + site_y
        )
        for well_x, row in enumerate('BCD')
        for well_y, col in enumerate('02 03 04'.split())
        for site_x in [1, 2, 3]
        for site_y in [1, 2, 3]
    ]
)

example2 = Viewer(
    [
        View(barcode='P101337', well='B02'),
        View(barcode='P101337', well='B03'),
        View(barcode='P101337', well='B04'),
        View(barcode='P101337', well='B05'),
        View(barcode='P101337', well='B06'),
        View(barcode='P101337', well='B07'),
        View(barcode='P101337', well='B08'),
        View(barcode='P101337', well='B09'),
        View(barcode='P101337', well='B10'),
    ]
)

example3 = Viewer(
    [
        View(barcode='P101337', well='B02', site=8, clip=ClipSquare(x, y, 150))
        for x in range(250, 2500, 500)
        for y in range(250, 2500, 500)
    ]
)

example4 = Viewer(
    [
        View(
            barcode='P101337', well='B02', site=8,
            clip=
                ClipBox(x, y, 144, 89)
                if (x + y) // 500 % 2 else
                ClipBox(x, y, 89, 144)
        )
        for x in range(250, 2500, 500)
        for y in range(250, 2500, 500)
    ]
)

style5 = 'padding: 1px 25px; border-radius: 10px; margin: 100px; background: #fff9; color: black; font-size: 20px; font-weight: 900'

example5 = Viewer(
    [
        View(barcode='P101337', well='B02', site=1, overlay='1', overlay_style=style5, overlay_dir='SE'),
        View(barcode='P101337', well='B02', site=2, overlay='2', overlay_style=style5, overlay_dir='S'),
        View(barcode='P101337', well='B02', site=3, overlay='3', overlay_style=style5, overlay_dir='SW'),
        View(barcode='P101337', well='B02', site=4, overlay='4', overlay_style=style5, overlay_dir='E'),
        View(barcode='P101337', well='B02', site=5, overlay='5', overlay_style=style5, overlay_dir='C'),
        View(barcode='P101337', well='B02', site=6, overlay='6', overlay_style=style5, overlay_dir='W'),
        View(barcode='P101337', well='B02', site=7, overlay='7', overlay_style=style5, overlay_dir='NE'),
        View(barcode='P101337', well='B02', site=8, overlay='8', overlay_style=style5, overlay_dir='N'),
        View(barcode='P101337', well='B02', site=9, overlay='9', overlay_style=style5, overlay_dir='NW'),
    ]
)

style6 = 'padding: 10px 25px; font-size: 20px; font-weight: 900; background: black'
cmpds = 'dmso fenb etop berb tetr flup sorb iono hexi'.split()
tab10 = '#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #e377c2 #7f7f7f #bcbd22 #17becf'.split()

example6 = Viewer(
    [
        View(barcode='P101337', well='B02', site=1, overlay=cmpds[0], overlay_style=f'{style6}; color: {tab10[0]}', hover=cmpds[0]),
        View(barcode='P101337', well='B03', site=2, overlay=cmpds[1], overlay_style=f'{style6}; color: {tab10[1]}', hover=cmpds[1]),
        View(barcode='P101337', well='B04', site=3, overlay=cmpds[2], overlay_style=f'{style6}; color: {tab10[2]}', hover=cmpds[2]),
        View(barcode='P101337', well='C02', site=4, overlay=cmpds[3], overlay_style=f'{style6}; color: {tab10[3]}', hover=cmpds[3]),
        View(barcode='P101337', well='C03', site=5, overlay=cmpds[4], overlay_style=f'{style6}; color: {tab10[4]}', hover=cmpds[4]),
        View(barcode='P101337', well='C04', site=6, overlay=cmpds[5], overlay_style=f'{style6}; color: {tab10[5]}', hover=cmpds[5]),
        View(barcode='P101337', well='D02', site=7, overlay=cmpds[6], overlay_style=f'{style6}; color: {tab10[6]}', hover=cmpds[6]),
        View(barcode='P101337', well='D03', site=8, overlay=cmpds[7], overlay_style=f'{style6}; color: {tab10[7]}', hover=cmpds[7]),
        View(barcode='P101337', well='D04', site=9, overlay=cmpds[8], overlay_style=f'{style6}; color: {tab10[8]}', hover=cmpds[8]),
    ]
)

def table(grid):
    res = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            assert isinstance(cell, View), f'Execpted View, got: {cell} ({y=}, {x=})'
            res += [replace(cell, x=x, y=y)]
    return Viewer(res)

style7 = 'background: black; padding: 0 10px'

example7 = table(
    [
        [
            View(hover='(padding at top left)'),
            View(overlay='site 1', overlay_style=style7),
            View(overlay='site 2', overlay_style=style7),
            View(overlay='site 3', overlay_style=style7),
            View(hover='(padding at top right)'),
        ],
        [
            View(overlay=cmpds[0], overlay_style=style7, overlay_dir='E'),
            View(barcode='P101337', well='B02', site=1),
            View(barcode='P101337', well='B02', site=2),
            View(barcode='P101337', well='B02', site=3),
        ],
        [
            View(overlay=cmpds[1], overlay_style=style7, overlay_dir='E'),
            View(barcode='P101337', well='B03', site=1),
            View(barcode='P101337', well='B03', site=2),
            View(barcode='P101337', well='B03', site=3),
        ],
        [
            View(overlay=cmpds[2], overlay_style=style7, overlay_dir='E'),
            View(barcode='P101337', well='B04', site=1),
            View(barcode='P101337', well='B04', site=2),
            View(barcode='P101337', well='B04', site=3),
        ],
        [
            View(hover='(padding at bottom left)'),
        ],
    ]
)

examples = [
    example1,
    example2,
    example3,
    example4,
    example5,
    example6,
    example7,
]

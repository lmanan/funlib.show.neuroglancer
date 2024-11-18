from .scale_pyramid import ScalePyramid
import neuroglancer


embeddings_shader_code = """
const float PI = 3.14159265;
struct MagPhi{
	float r;
  	float theta;
};

struct HSV{
  float h;
  float s;
  float v;
};

struct RGB{
  float r;
  float g;
  float b;
};


RGB hsv2rgb(HSV hsv){
  // h should be between 0.0 degrees and 360 degrees
  // s should be between 0.0 and 1.0
  // v should be between 0.0 and 1.0
  float c = (hsv.s)*(hsv.v);
  float hp = hsv.h/60.0;



  float t =abs(mod(hp,2.0)-1.0);
  float x = c*(1.0- t);
  float r1, g1, b1;
  if(0.0 <= hp && hp < 1.0){
     r1 = c;
     g1 = x;
     b1 = 0.0;
  }else if (1.0<= hp && hp <2.0){
    r1 = x;
    g1 = c;
    b1 = 0.0;
  }else if (2.0<= hp && hp <3.0){
    r1 = 0.0;
    g1 = c;
    b1 = x;
  }else if (3.0<= hp && hp <4.0){
    r1 = 0.0;
    g1 = x;
    b1 = c;
  }else if (4.0<= hp && hp <5.0){
    r1 = x;
    g1 = 0.0;
    b1 = c;
  }else if (5.0<= hp && hp <6.0){
    r1 = c;
    g1 = 0.0;
    b1 = x;
  }else{
  	r1 = 0.0;
    g1 = 0.0;
    b1 = 0.0;
  }
  float m = hsv.v - c;
  RGB rgb;
  rgb.r = r1+m;
  rgb.g = g1+m;
  rgb.b = b1+m;
  return rgb;
}


MagPhi cart2polar(float o_x, float o_y)
{
  MagPhi magphi;
  magphi.r = sqrt(o_x*o_x + o_y*o_y);
  float theta;


  if (o_x >0.0){
    theta = atan(o_y/o_x);
  }else{
  	theta = atan(o_y/o_x) + PI;
  }
  if (theta<0.0){
  	magphi.theta = 2.0*PI+theta;
  }else{
  	magphi.theta = theta;
  }
  return magphi;}



void main() {
    MagPhi magphi = cart2polar(getDataValue(%i), getDataValue(%i));
	HSV hsv;
  	hsv.h = magphi.theta * 180.0/PI;
  	hsv.s = magphi.r/25.0;
    hsv.v = max(1.0 - getDataValue(%i)/10.0, 0.0);

    RGB rgb = hsv2rgb(hsv);
  	emitRGB(1.0*vec3(
      rgb.r,
      rgb.g,
      rgb.b)
      );
}"""


rgb_shader_code = """
void main() {
    emitRGB(
        %f*vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}"""

color_shader_code = """
void main() {
    emitRGBA(
        vec4(
        %f, %f, %f,
        toNormalized(getDataValue()))
        );
}"""

binary_shader_code = """
void main() {
  emitGrayscale(255.0*toNormalized(getDataValue()));
}"""

heatmap_shader_code = """
void main() {
    float v = toNormalized(getDataValue(0));
    vec4 rgba = vec4(0,0,0,0);
    if (v != 0.0) {
        rgba = vec4(colormapJet(v), 1.0);
    }
    emitRGBA(rgba);
}"""


def parse_dims(array):

    if type(array) == list:
        array = array[0]

    dims = len(array.data.shape)
    spatial_dims = array.roi.dims
    channel_dims = dims - spatial_dims

    print("dims        :", dims)
    print("spatial dims:", spatial_dims)
    print("channel dims:", channel_dims)

    return dims, spatial_dims, channel_dims


def create_coordinate_space(array, spatial_dim_names, channel_dim_names, unit):

    dims, spatial_dims, channel_dims = parse_dims(array)
    assert spatial_dims > 0

    if channel_dims > 0:
        channel_names = channel_dim_names[-channel_dims:]
    else:
        channel_names = []
    spatial_names = spatial_dim_names[-spatial_dims:]
    names = channel_names + spatial_names
    units = [""] * channel_dims + [unit] * spatial_dims
    scales = [1] * channel_dims + list(array.voxel_size)

    print("Names    :", names)
    print("Units    :", units)
    print("Scales   :", scales)

    return neuroglancer.CoordinateSpace(names=names, units=units, scales=scales)


def create_shader_code(
    shader, channel_dims, rgb_channels=None, color=None, scale_factor=1.0, name=None
):

    if shader is None:
        if 'embeddings' in name:
            shader = "embeddings"
        else:
            if channel_dims > 1:
                shader = "rgb"
            else:
                return None

    if rgb_channels is None:
        rgb_channels = [0, 1, 2]

    if shader == "embeddings":
        return embeddings_shader_code % (
            rgb_channels[0],
            rgb_channels[1],
            rgb_channels[2],
        )
    if shader == "rgb":
        return rgb_shader_code % (
            scale_factor,
            rgb_channels[0],
            rgb_channels[1],
            rgb_channels[2],
        )

    if shader == "color":
        assert (
            color is not None
        ), "You have to pass argument 'color' to use the color shader"
        return color_shader_code % (
            color[0],
            color[1],
            color[2],
        )

    if shader == "binary":
        return binary_shader_code

    if shader == "heatmap":
        return heatmap_shader_code


def add_layer(
    context,
    array,
    name,
    spatial_dim_names=None,
    channel_dim_names=None,
    opacity=None,
    shader=None,
    rgb_channels=None,
    color=None,
    visible=True,
    value_scale_factor=1.0,
    units="nm",
):
    """Add a layer to a neuroglancer context.

    Args:

        context:

            The neuroglancer context to add a layer to, as obtained by
            ``viewer.txn()``.

        array:

            A ``daisy``-like array, containing attributes ``roi``,
            ``voxel_size``, and ``data``. If a list of arrays is given, a
            ``ScalePyramid`` layer is generated.

        name:

            The name of the layer.

        spatial_dim_names:

            The names of the spatial dimensions. Defaults to ``['t', 'z', 'y',
            'x']``. The last elements of this list will be used (e.g., if your
            data is 2D, the channels will be ``['y', 'x']``).

        channel_dim_names:

            The names of the non-spatial (channel) dimensions. Defaults to
            ``['b^', 'c^']``.  The last elements of this list will be used
            (e.g., if your data is 2D but the shape of the array is 3D, the
            channels will be ``['c^']``).

        opacity:

            A float to define the layer opacity between 0 and 1.

        shader:

            A string to be used as the shader. Possible values are:

                None     :  neuroglancer's default shader
                'rgb'    :  An RGB shader on dimension `'c^'`. See argument
                            ``rgb_channels``.
                'color'  :  Shows intensities as a constant color. See argument
                            ``color``.
                'binary' :  Shows a binary image as black/white.
                'heatmap':  Shows an intensity image as a jet color map.

        rgb_channels:

            Which channels to use for RGB (default is ``[0, 1, 2]``).

        color:

            A list of floats representing the RGB values for the constant color
            shader.

        visible:

            A bool which defines the initial layer visibility.

        value_scale_factor:

            A float to scale array values with for visualization.

        units:

            The units used for resolution and offset.
    """

    if channel_dim_names is None:
        channel_dim_names = ["b", "c^"]
    if spatial_dim_names is None:
        spatial_dim_names = ["t", "z", "y", "x"]

    if rgb_channels is None:
        rgb_channels = [0, 1, 2]

    is_multiscale = type(array) == list

    dims, spatial_dims, channel_dims = parse_dims(array)

    if is_multiscale:

        dimensions = []
        for a in array:
            dimensions.append(
                create_coordinate_space(a, spatial_dim_names, channel_dim_names, units)
            )

        # why only one offset, shouldn't that be a list?
        voxel_offset = [0] * channel_dims + list(
            array[0].roi.offset / array[0].voxel_size
        )

        layer = ScalePyramid(
            [
                neuroglancer.LocalVolume(
                    data=a.data, voxel_offset=voxel_offset, dimensions=array_dims
                )
                for a, array_dims in zip(array, dimensions)
            ]
        )

    else:

        voxel_offset = [0] * channel_dims + list(array.roi.offset / array.voxel_size)

        dimensions = create_coordinate_space(
            array, spatial_dim_names, channel_dim_names, units
        )

        layer = neuroglancer.LocalVolume(
            data=array.data,
            voxel_offset=voxel_offset,
            dimensions=dimensions,
        )

    shader_code = create_shader_code(
        shader, channel_dims, rgb_channels, color, value_scale_factor, name=name
    )

    if opacity is not None:
        if shader_code is None:
            context.layers.append(
                name=name, layer=layer, visible=visible, opacity=opacity
            )
        else:
            context.layers.append(
                name=name,
                layer=layer,
                visible=visible,
                shader=shader_code,
                opacity=opacity,
            )
    else:
        if shader_code is None:
            context.layers.append(name=name, layer=layer, visible=visible)
        else:
            context.layers.append(
                name=name, layer=layer, visible=visible, shader=shader_code
            )

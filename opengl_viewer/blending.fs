#version 330 core
layout(location = 0) out vec4 FragColor;
// out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture1;

void main()
{             
	vec4 rgba = texture(texture1, TexCoords);
    FragColor = vec4(rgba.xyz, rgba.w);
}
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform vec3 bco;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;

void main()
{   
	vec4 fc1 = texture(texture1, TexCoords);
	vec4 fc2 = texture(texture2, TexCoords);
	vec4 fc3 = texture(texture3, TexCoords);

	// fc1.xyz *= fc1.w;
	// fc2.xyz *= fc2.w;
	// fc3.xyz *= fc3.w;

	float alpha = (1./255 + fc1.w * bco.x + fc2.w * bco.y + fc3.w * bco.z);
	vec3 rgb = (fc1.xyz * bco.x + fc2.xyz * bco.y + fc3.xyz * bco.z) / alpha;
	// vec3 rgb = vec3(alpha,alpha,alpha);
	// vec3 rgb = (fc1.xyz + fc2.xyz) / (1e-8 + fc1.w + fc2.w);
	// vec3 rgb = fc1.xyz / (1e-8+fc1.w);
	// veg3 rgb = fc1.xyz / fc1.w;
	// vec3 rgb = (fc1.xyz + fc2.xyz + fc3.xyz) / (1e-8 + fc1.w + fc2.w + fc3.w);
	FragColor = vec4(rgb, 1.);

	// FragColor = fc1 * bco.x + fc2 * bco.y + fc3 * bco.z;

	// FragColor = vec4(bco, 1.);
	// FragColor = (fc1 + fc2 + fc3) / 3.;

    // FragColor = texture(texture2, TexCoords);
    // FragColor = (texture(texture1, TexCoords) + texture(texture2, TexCoords) + texture(texture3, TexCoords))/3.;
    // FragColor = vec4(TexCoords, 0.0, 1.0);
}
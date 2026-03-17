import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

#h를 통한 반사면 gradient 계산
def compute_surface_gradients(
    h:np.ndarray,
    dx: float,
    dy: float
) -> tuple:
    
    grad = np.gradient(h,dy,dx)
    dh_dy = grad[0]
    dh_dx = grad[1]
    return dh_dx, dh_dy


#입사 복소장 U_in(x,y)

def compute_U_in(
    X: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    A0: float,
    wavelength: float
) -> np.ndarray:
    k = 2.0 * np.pi / wavelength
    phase=k*(X+h)
    
    return A0 * np.exp(1j * phase)

# (x,y,z)에서의 입사각, z=x+h

def compute_incident_angle(
    dh_dx: np.ndarray,
    dh_dy: np.ndarray
) -> np.ndarray:
    nz=1.0
    norm_n=np.sqrt((1.0+dh_dx)**2+dh_dy**2+1.0)
    
    cos_theta_loc=np.clip(nz/norm_n, -1.0, 1.0)
    
    return np.arccos(cos_theta_loc)

#복소 프레넬 반사계수 R(x,y)

def compute_fresnel_reflection(
    theta_loc: np.ndarray,
    n1: float,
    n2_complex: complex
) ->np.ndarray:
    
    n2=complex(n2_complex)
    cos_loc=np.cos(theta_loc).astype(complex)
    sin_loc=np.sin(theta_loc).astype(complex)
    
    sin_t=(n1/n2)*sin_loc
    cos_t=np.sqrt(1.0-sin_t**2)
    
    r_s=(n1*cos_loc-n2*cos_t) / (n1*cos_loc+n2*cos_t)
    r_p=(n2*cos_loc-n1*cos_t)/(n2*cos_loc+n1*cos_t)
    
    return (r_s+r_p)/2.0


#반사 직후 복소장

def compute_U_ref(U_in: np.ndarray, R_tilde: np.ndarray) -> np.ndarray:
    return U_in*R_tilde

# Rayleigh-sommerfeld 적분

def compute_U_CMOS_loop(
    U_ref: np.ndarray,  #반사 직후 복소장
    X:np.ndarray,   #반사면 x좌표
    Y:np.ndarray,   #반사면 y좌표
    h:np.ndarray,   #반사면 높이
    dh_dx_ref: np.ndarray,  #반사면 편미분
    dh_dy_ref: np.ndarray,  #반사면 편미분
    Y_prime: np.ndarray, #cmos y좌표
    Z_prime: np.ndarray, #cmos z좌표
    wavelength: float,  #레이저 파장
    x_cmos_location: float, #cmos 평면 x좌표
    dx_ref_grid: float, #반사면 x그리드 간격
    dy_ref_grid: float  #반사면 y그리드 간격
    
) -> np.ndarray:
    k=2.0*np.pi/wavelength
    Ny, Nx = U_ref.shape
    Nz_cmos, Ny_cmos=Y_prime.shape
    
    U_CMOS=np.zeros((Nz_cmos, Ny_cmos), dtype=complex)
    
    for j_cmos in range(Nz_cmos):   #cmos z 인덱스
        for i_cmos in range(Ny_cmos):   #cmos y인덱스
            y_p=Y_prime[j_cmos,i_cmos]   #cmos y좌표
            z_p=Z_prime[j_cmos,i_cmos]   #cmos z좌표
            
            integral=0.0+0.0j
            
            for j in range(Ny):
                for i in range(Nx):
                    x_s=X[j,i]  #반사면 x좌표
                    y_s=Y[j,i]
                    h_s=h[j,i]
                    u_s=U_ref[j,i]  #반사면 해당 좌표에서 복소장
                    dhx=dh_dx_ref[j,i]  #반사면 해당 좌표에서 편미분
                    dhy=dh_dy_ref[j,i]
                    
                    delta_x=x_cmos_location-x_s #반사면과 CMOS 사이 x거리
                    delta_y=y_p-y_s
                    delta_z=z_p-(x_s+h_s)
                    
                    r=np.sqrt(delta_x**2+delta_y**2+delta_z**2)
                    
                    obliquity=(1.0+dhx)*delta_x+dhy*delta_y-delta_z
                    
                    integrand=u_s*(np.exp(1j*k*r)/r**2)*obliquity*dx_ref_grid*dy_ref_grid
                    integral += integrand
                    
                    
            U_CMOS[j_cmos,i_cmos]=integral/(1j*wavelength)
            
    return U_CMOS

def forward_propagate(
    h:np.ndarray,   #변형 높이
    x_coords: np.ndarray,   #반사면 좌표
    y_coords: np.ndarray,   
    y_prime_coords: np.ndarray,    #cmos좌표
    z_prime_coords: np.ndarray,
    wavelength: float,
    A0: float,
    n1: float,
    n2_complex: complex,
    x_cmos_location: float
) -> dict:
    
    dx=float(x_coords[1]-x_coords[0])   #반사면 그리드 간격
    dy=float(y_coords[1]-y_coords[0])
    
    X,Y=np.meshgrid(x_coords, y_coords) #반사면 메쉬 size는(Ny, Nx)
    Y_prime, Z_prime=np.meshgrid(y_prime_coords, z_prime_coords)    #CMOS 메쉬
    
    dh_dx, dh_dy = compute_surface_gradients(h,dx,dy)   #표면 편미분
    
    U_in=compute_U_in(X,Y,h,A0,wavelength)  #거울 들어가는 입사광 복소장
    
    theta_loc=compute_incident_angle(dh_dx, dh_dy)  #입사광의 입사각
    
    R_tilde=compute_fresnel_reflection(theta_loc, n1, n2_complex)   #Fresnel 반사계수
    
    U_ref=compute_U_ref(U_in,R_tilde)   #반수 직후 복소장
    
    U_CMOS=compute_U_CMOS_loop(
        U_ref, X,Y,h,dh_dx,dh_dy,Y_prime,Z_prime,wavelength,x_cmos_location,dx,dy
    )   #rs전파
    
    
    return{
        'U_in': U_in,
        'theta_loc': theta_loc,
        'R_tilde': R_tilde,
        'U_ref': U_ref,
        'U_CMOS': U_CMOS,
        'I_CMOS': np.abs(U_CMOS)**2,
        'dh_dx':dh_dx,
        'dh_dy':dh_dy
    }
    

if __name__=="__main__":
    import time
    
    #레이저, 매질 파라미터
    wavelength=500e-9
    A0=1.0
    n1=1.0
    n2_complex=complex(0.97112,1.8737)    #500nm에서의 금 복소굴절률인데, 나중에 맞춰서 수정
    
    #반사면 그리드
    pixel_size=4e-6
    N=16    #해상도 이후 수정
    
    x_coords=np.linspace(1,N,N)*pixel_size #x>0
    y_coords=np.linspace(-N//2, N//2-1,N)*pixel_size
    
    #표면 높이맵
    X_tmp, Y_tmp=np.meshgrid(x_coords, y_coords)
    sigma=3*pixel_size
    h_max=500e-6
    h=h_max*np.exp(
        -((X_tmp-x_coords[n//2])**2+Y_tmp**2)/(2*sigma**2)
    )
    
    x_cmos_location=x_coords[-1]+5e-3   #반사면 오른쪽 5mm에 cmos배치
    
    y_prime_coords=y_coords.copy()  #cmos넓이가 반사면 넓이랑 같게 설정했는데 이후 수정 가능
    z_prime_coords=x_coords.copy()
    
    result_loop=forward_propagate(
        h, x_coords, y_coords, y_prime_coords, z_prime_coords, wavelength,
        A0, n1, n2_complex, x_cmos_location
    )
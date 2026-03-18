import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from tqdm import tqdm
from skimage.restoration import unwrap_phase

#h를 통한 반사면 gradient 계산
def compute_surface_gradients(
    h:np.ndarray,
    x:np.ndarray,
    y:np.ndarray
) -> tuple:
    
    grad = np.gradient(h,y,x)
    dh_dy = grad[0]
    dh_dx = grad[1]
    return dh_dx, dh_dy


#입사 복소장 U_in(x,y)

def compute_U_in(
    X: np.ndarray,
    h: np.ndarray,
    t: float,
    A0: float,
    wavelength: float
) -> np.ndarray:
    k = 2.0 * np.pi / wavelength
    phase=k*(X+h-t)
    
    return (A0 * np.exp(1j * phase)).astype(complex)

# (x,y,z)에서의 입사각, z=x+h

def compute_incident_angle(
    dh_dx: np.ndarray,
    dh_dy: np.ndarray
) -> np.ndarray:
    nz=1.0
    norm_n=np.sqrt((1.0+dh_dx)**2+dh_dy**2+1.0)
    
    cos_theta_loc=np.clip(1.0/norm_n, -1.0, 1.0)
    
    return np.arccos(cos_theta_loc)

#복소 프레넬 반사계수 R(x,y)

def compute_fresnel_reflection(
    theta_loc: np.ndarray,
    n1: float,
    n2_complex: complex
) ->np.ndarray:
    
    n2=complex(n2_complex)
    cos_i=np.cos(theta_loc).astype(complex)
    sin_i=np.sin(theta_loc).astype(complex)
    
    sin_t=(n1/n2)*sin_i
    cos_t=np.sqrt(1.0-sin_t**2)
    
    r_s=(n1*cos_i-n2*cos_t) / (n1*cos_i+n2*cos_t)
    r_p=(n2*cos_i-n1*cos_t)/(n2*cos_i+n1*cos_t)
    
    return (r_s+r_p)/2.0


#반사 직후 복소장

def compute_U_ref(U_in: np.ndarray, R_tilde: np.ndarray) -> np.ndarray:
    return (U_in*R_tilde).astype(complex)

# Rayleigh-sommerfeld 적분

def compute_U_CMOS_loop(
    U_ref: np.ndarray,  #반사 직후 복소장
    X:np.ndarray,   #반사면 x좌표
    Y:np.ndarray,   #반사면 y좌표
    h:np.ndarray,   #반사면 높이
    dh_dx_ref: np.ndarray,  #반사면 편미분
    dh_dy_ref: np.ndarray,  #반사면 편미분
    Y_prime_coords: np.ndarray, #cmos y좌표
    Z_prime_coords: np.ndarray, #cmos z좌표
    wavelength: float,  #레이저 파장
    x_cmos_location: float, #cmos 평면 x좌표
    dx_ref_grid: float, #반사면 x그리드 간격
    dy_ref_grid: float  #반사면 y그리드 간격
    
) -> np.ndarray:
    k_wave=2.0*np.pi/wavelength
    
    Nz_cmos=len(z_prime_coords)
    Ny_cmos=len(y_prime_coords)
    Z_s=X+h
    
    U_CMOS=np.zeros((Nz_cmos, Ny_cmos),dtype=complex)
    
    pbar = tqdm(
        total=Nz_cmos * Ny_cmos,
        desc="RS integral",
        unit="px",
        bar_format=("{desc}: {percentage:3.0f}%|{bar:30}| "
                    "{n_fmt}/{total_fmt} px  [{elapsed}<{remaining},  {rate_fmt}]"),
        colour="cyan",
        dynamic_ncols=True,
    )
    
    with pbar:
        for j_cmos in range(Nz_cmos):   # CMOS z 인덱스
            for i_cmos in range(Ny_cmos):   # CMOS y 인덱스

                z_p = z_prime_coords[j_cmos]    #cmos z좌표
                y_p = y_prime_coords[i_cmos]    #cmos y좌표

                delta_x   = x_cmos_location - X     #반사면과 cmos사이 거리
                delta_y   = y_p             - Y     #cmos y좌표와 반사면 y좌표 사이 거리
                delta_z   = z_p             - Z_s

                r         = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                obliquity = (1.0 + dh_dx_ref)*delta_x + dh_dy_ref*delta_y - delta_z
                integrand = U_ref * (np.exp(1j*k_wave*r) / r**2) * obliquity \
                            * dx_ref_grid * dy_ref_grid

                U_CMOS[j_cmos, i_cmos] = np.sum(integrand) / (1j * wavelength)
                pbar.update(1)

    return U_CMOS
    

#rs 전파

def forward_propagate(
    h:np.ndarray,   #변형 높이
    x_coords: np.ndarray,   #반사면 좌표
    y_coords: np.ndarray,   
    y_prime_coords: np.ndarray,    #cmos좌표
    z_prime_coords: np.ndarray,
    wavelength: float,
    A0: float,
    t: float,
    n1: float,
    n2_complex: complex,
    x_cmos_location: float
) -> dict:
    
    dx_ref_grid=float(x_coords[1]-x_coords[0])   #반사면 그리드 간격
    dy_ref_grid=float(y_coords[1]-y_coords[0])
    
    X,Y=np.meshgrid(x_coords, y_coords) #반사면 좌표->그리드

    dh_dx, dh_dy = compute_surface_gradients(h,x_coords,y_coords)   #표면 편미분
    
    U_in=compute_U_in(X,Y,h,A0,wavelength)  #거울 들어가는 입사광 복소장
    
    theta_loc=compute_incident_angle(dh_dx, dh_dy)  #입사광의 입사각
    
    R_tilde=compute_fresnel_reflection(theta_loc, n1, n2_complex)   #Fresnel 반사계수
    
    U_ref=compute_U_ref(U_in,R_tilde)   #반수 직후 복소장
    
    U_CMOS=compute_U_CMOS_loop(
        U_ref, X,Y,h,dh_dx,dh_dy,
        y_prime_coords,z_prime_coords,
        wavelength, x_cmos_location,
        dx_ref_grid, dy_ref_grid
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
    t=0.0
    n1=1.0
    n2_complex=complex(0.97112,1.8737)    #500nm에서의 금 복소굴절률인데, 나중에 맞춰서 수정
    
    #반사면 그리드
    pixel_size=3e-6
    N=3000    #픽셀 개수
    
    x_coords=np.linspace(1,N,N)*pixel_size #x>0
    y_coords=np.linspace(-N//2, N//2-1,N)*pixel_size
    x_coords_center=x_coords[N//2]
    y_coords_center=y_coords[N//2]
    
    #표면 높이맵
    X_tmp, Y_tmp=np.meshgrid(x_coords, y_coords)
    
    h_amplitude = -wavelength / 8     # h 최대 높이
    h_sigma     = 5 * pixel_size      # 가우시안 반경
    h_cx        = x_coords[N//2]      # 가우시안 x중심
    h_cy        = 0.0                 # 가우시안 y중심

    h_def = h_amplitude * np.exp(
        -((X_tmp - h_cx)**2 + (Y_tmp - h_cy)**2) / (2 * h_sigma**2)
    )
    
    h_ref = np.zeros_like(h_def)    # h=0반사면
    
    x_cmos_location=x_coords[-1]+10e-3   #반사면 오른쪽 10mm에 cmos배치
    N_cmos = 1000   #cmos 픽셀 개수
    pixel_size_cmos=3e-6    #cmos 픽셀 사이즈
    y_prime_coords=np.linspace(y_coords_center-N_cmos//2,y_coords_center+N_cmos//2,N_cmos)
    z_prime_coords=np.linspace(x_coords_center-N_cmos//2,x_coords_center+N_cmos//2,N_cmos)

    
    result_loop=forward_propagate(
        h_def, x_coords, y_coords, y_prime_coords, z_prime_coords, wavelength,
        A0, t, n1, n2_complex, x_cmos_location
    )
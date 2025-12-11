import numpy as np
import legume
import autograd.numpy as npa
from legume import backend as bd
import json
import base64
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.cm as cm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable



def NG(gme,kind,mode,height=None,Nx=100,Ny=125):
    """
    Calculate the group index (ng) for a specific mode and k-point in a photonic crystal waveguide.
    
    This function computes the group index by calculating the Poynting vector from the electric 
    and magnetic fields at a specific k-point and mode index. The group index is proportional to 
    the inverse of the energy velocity in the structure.
    
    Args:
        gme: Guided mode expansion object containing the photonic crystal information
        kind: Index of the k-point to use for calculation
        mode: Index of the mode to use for calculation
        height: Height within the slab to evaluate the fields (defaults to middle of slab)
        Nx: Number of points in x-direction for field calculation
        Ny: Number of points in y-direction for field calculation
        
    Returns:
        ng: Group index value
    """

    if height is None:
        height = gme.phc.layers[0].d/2

    Efield,_,_ = gme.get_field_xy('E',kind,mode,height,Nx=Nx,Ny=Ny)
    Hfield,_,_ = gme.get_field_xy('H',kind,mode,height,Nx=Nx,Ny=Ny)
    Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
    ng = -1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*gme.phc.lattice.a2[1]/Nx/Ny*gme.phc.layers[0].d)
    return(ng)


def hole_borders(phc,phidiv=45):
    """
    returns the points at the hole borders
    """

    #get relevent information for computation
    shapes = phc.layers[0].shapes
    phis = npa.linspace(0,2*npa.pi,phidiv,endpoint=False)
    cphis = bd.cos(phis); sphis = bd.sin(phis)

    #get the array of hole atributes [xpos,ypos,r]
    holeCords = bd.array([[s.x_cent, s.y_cent, s.r] for s in shapes])

    #get the coordinates of the hole borders
    # Compute the initial borders using broadcasting
    x_coords = holeCords[:, 0:1] + holeCords[:, 2:3] * cphis  # Add dimensions for broadcasting
    y_coords = holeCords[:, 1:2] + holeCords[:, 2:3] * sphis

    # Combine x and y coordinates
    initialBorders = bd.stack([x_coords, y_coords], axis=-1)

    # Apply lattice corrections
    xCorrected = bd.where(initialBorders[..., 0] > phc.lattice.a1[0] / 2, 
                            initialBorders[..., 0] - phc.lattice.a1[0], 
                            initialBorders[..., 0])
    yCorrected = bd.where(initialBorders[..., 1] > phc.lattice.a2[0] / 2, 
                            initialBorders[..., 1] - phc.lattice.a2[0], 
                            initialBorders[..., 1])

    # Combine corrected coordinates
    borders = bd.stack([xCorrected, yCorrected], axis=-1)

    return(borders, phis, holeCords[:,2])


def get_xyfield(gme,phc,n,xys,k,zdiv=1,field='E'):
    """
    returns the field around the holes

    Args:
        gme : the GuidedModeExp object
        n : the mode index
        phc : the photonic crystal object
        xys : the coordinates of the points to get the field at, shape (:,:,2)
        z : the z-coordinate of the points to get the field at
        field : the field to get
        components : the components of the field to get
    """

    zSpace = npa.linspace(0,phc.layers[0].d,zdiv+1)
    zs = (zSpace[1:]+zSpace[:-1])/2

    FT = bd.stack(
        [
            bd.stack(gme.ft_field_xy(field, k, n, z), axis=0)
            for z in zs
        ],
        axis=0)

    phase = bd.exp(1j*bd.matmul(xys,gme.gvec))

    fis = npa.einsum('ijk,abk->abij',phase,FT)
    
    return(fis)


def comp_pdote(gme,phc,n,borders,phis,k,zdiv=1):
    """
    Computes the E and D dot products for the alpha calculation
    """

    E = get_xyfield(gme,phc,n,borders,k,zdiv=zdiv)
    D = get_xyfield(gme,phc,n,borders,k,zdiv=zdiv,field='D')

    Epara = bd.array([-bd.sin(phis)*E[:,0],bd.cos(phis)*E[:,1]])
    Dperp = bd.array([bd.cos(phis)*D[:,0],bd.sin(phis)*D[:,1]])

    p = Epara+(phc.layers[0].eps_b+1)*Dperp/(2*phc.layers[0].eps_b*1)

    pdeR = bd.conj(E[:,0])*bd.conj(p[0])+bd.conj(E[:,1])*bd.conj(p[1])
    pdeRP = E[:,0]*p[0]+E[:,1]*p[1]

    return(pdeR,pdeRP)

def comp_backscatter(gme, phc, n, k, a=266, sig=3, lp=40, phidiv=45, zdiv=10):
    """
    This runs the calculation of the backscattering divided by the group index
    Given the simulation results

    returns the backscattering loss in units of a^-1
    """
    # get the points around the hole
    borders, phis, holeRad = hole_borders(phc,phidiv=phidiv)

    # process phis so that they work with the formula
    phisLooped = npa.arctan(npa.tan(phis))

    # get the necessary field information around the holes
    pdeR, pdeRP = comp_pdote(gme, phc, n, borders, phis, k, zdiv=zdiv)

    # Efficiently compute p dot e cross products using broadcasting
    # pdeR.shape = (Nholes, Nphis), pdeRP.shape = (Nholes, Nphis)
    # Compute meshgrid for each hole at once
    pdeR_ = pdeR[:,:, :, None]  # (zs, Nholes, Nphis, 1)
    pdeRP_ = pdeRP[:, :, None, :]  # (zs, Nholes, 1, Nphis)
    preSumPde = pdeR_ * pdeRP_  # (zs, Nholes, Nphis, Nphis)

    # the real exponential term, using broadcasting
    phiMesh = phisLooped[None, :, None]  # (1, Nphis, 1)
    phiPMesh = phisLooped[None, None, :] # (1, 1, Nphis)
    phiDiff = phiMesh - phiPMesh          # (1, Nphis, Nphis)
    realExp = (npa.abs(phiDiff) * (-holeRad)[:, None, None]) / (lp / a)

    # the imaginary exponential term, using broadcasting
    x = borders[:, :, 0]  # (Nholes, Nphis)
    xMesh = x[:, :, None]     # (Nholes, Nphis, 1)
    xPMesh = x[:, None, :]    # (Nholes, 1, Nphis)
    kx = 2 * npa.linalg.norm(gme.kpoints[:, 0])
    imagExp = kx * (xMesh - xPMesh)  # (Nholes, Nphis, Nphis)

    # run the integral, including the jacobian determinant
    intigrand = preSumPde * npa.exp(realExp[None,:,:,:] + 1j * imagExp[None,:,:,:])
    weights = (holeRad * npa.pi * 2 / phidiv) ** 2  # shape: (Nholes,)
    intigral = npa.sum(intigrand, axis=(2, 3)) * weights

    # calculate the leading coefficients for each of the holes
    cirleCoeffs = ((299792458 * 2 * npa.pi * gme.freqs[0, n]) * (sig / a) * (phc.layers[0].eps_b - 1) / 2) ** 2

    # compute the final result
    alpha = bd.real(cirleCoeffs * npa.sum(intigral) * (phc.layers[0].d * a * 1e-9) ** 2 / zdiv)
    
    return(alpha*266*1E-9) #this puts it in units of a^-1

def backscatterLog(gme,phc,n,k=0,a=266,sig=3,lp=40,phidiv=45,zdiv=1):
    """
    returns the cost associated with the backscattering
    """
    alpha = npa.log10(comp_backscatter(gme,phc,n,k,a,sig,lp,phidiv,zdiv))

    return(alpha)

def dispLossPlot(vars,crystal,kpoints,path,gmax=4.01,phcParams={},mode=14,a=455,final_cost=1e9,execution_time=0,niter=0):
    """
        Saves a figure with key information including the ng, dispersion, loss, and loss/ng^2

        Args:
            vars: optimal parameters from the minimization
            crystal: the type of crystal to use
            kpoints: kpoints to plot
            path: path to save the figure
            gmax: maximum group index
            phcParams: parameters for the photonic crystal
            mode: mode to plot
            a: lattice constant
            final_cost: final cost of the minimization

        Returns:
            None
    """
    text_scale = 0.85

    #find the k-points that we optimized for 
    ks = np.linspace(np.pi*.5,np.pi,200)
    kind = [int(np.argmin(np.abs(k-ks))) for k in kpoints]

    #run GME 
    gmeParams = {'verbose':False,'numeig':mode+2,'compute_im':False,'kpoints':np.vstack((ks,[0]*len(ks)))}
    phc = crystal(vars,**phcParams)
    gme = legume.GuidedModeExp(phc,gmax)
    gme.run(**gmeParams)    

    #run ng and loss
    ng = []
    loss = []
    for i in range(len(ks)):
        ng.append(np.abs(NG(gme,i,mode,Nx=100,Ny=125)))
        loss.append(10**backscatterLog(gme,phc,mode,k=i,a=a,zdiv=10)/a/1E-7*10*np.log10(np.e)*ng[-1]**2)
    ng = np.array(ng)
    loss = np.array(loss)
    
    # Gridspec setup: [dispersion+ng+loss] | [empty] | [field imshow] | [side table]
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[2, .75, .5, 1.05],
        wspace=0.22
    )

    # ---------------------- Dispersion and Data Extraction ----------------------
    wl = 1 / (gme.freqs[:, 14] / a)  # Wavelength (nm)
    wl_band = wl[kind[0]:kind[-1]+1]
    min_wl, max_wl = wl[kind[0]], wl[kind[-1]]
    wl_range = max_wl - min_wl

    fs_hz = gme.freqs[:, 14] * 299792458 / a / 1e-9  # [Hz]

    # Dispersion
    disp = (
        1 / (2 * np.pi * 299792458)
        * (np.array(ng[1:]) - np.array(ng[:-1]))
        / (fs_hz[1:] - fs_hz[:-1])
        * 1e24
    )

    # Determine bandwidth, light line
    light_line_idx = int(np.argmin(np.abs(gme.freqs[:,14] - ks/np.pi/2)))
    light_line_wl = 1/(gme.freqs[light_line_idx, 14]/a)
    bw_min = 1/(max(np.max(gme.freqs[:,14-1]), np.min(gme.freqs[:,14]))/a)
    bw_max = 1/(min(np.min(gme.freqs[:,14+1]), gme.freqs[light_line_idx,14])/a)

    # ---------------------- Main Plot: Dispersion Curve ------------------------
    ax = fig.add_subplot(gs[0])

    # Dispersion
    ax.plot(
        wl[:-1], np.abs(disp),
        color='tab:blue', label='Dispersion', linewidth=1.5
    )
    ax.set_xlabel('Wavelength (nm)', fontsize=12 * text_scale)
    ax.set_ylabel(r'Dispersion [ps$^2$/m]', color='tab:blue', fontsize=11.5 * text_scale)
    ax.set_yscale('log')
    ax.set_ylim(1e1, 1e6)
    ax.set_xlim(min_wl - wl_range * 0.75, max_wl + wl_range * 0.75)
    ax.tick_params(axis='both', which='major', labelsize=10 * text_scale, length=3)

    # Highlight regions
    ax.fill_between([bw_min, bw_max], 1e1, 1e6, color='lightblue', alpha=0.38)
    ax.fill_between([min_wl, max_wl], 1e1, 1e6, color='blue', alpha=0.19)
    ax.fill_between([min_wl - wl_range * 0.75, light_line_wl], 1e1, 1e6, color='gray', alpha=0.21)

    # ------------------------- Other Y Axes -------------------------
    # Group Index (ng)
    ax2 = ax.twinx()
    ax2.plot(
        wl, ng,
        color='tab:red', label='ng', linewidth=1.2
    )
    ax2.set_ylabel('Group Index ng', color='tab:red', fontsize=11.5 * text_scale)
    ax2.tick_params(axis='y', which='major', labelsize=10 * text_scale, length=3)
    ax2.set_ylim(5, 50)

    # Loss (dB/cm)
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("outward", 45))
    loss_visible = loss[kind[0]:kind[-1]+1]
    loss_range = np.ptp(loss_visible)
    ax3.plot(
        wl, loss,
        color='tab:green', label='Loss', linewidth=1.2, linestyle='--'
    )
    ax3.set_ylabel(r'Loss [dB/cm]', color='tab:green', fontsize=11.5 * text_scale)
    ax3.tick_params(axis='y', which='major', labelsize=10 * text_scale, length=3)
    ax3.set_ylim(
        np.min(loss_visible) - 0.25 * loss_range,
        np.max(loss_visible) + 0.25 * loss_range
    )

    # Loss / ng^2
    ax4 = ax.twinx()
    ax4.spines.right.set_position(("outward", 90))
    loss_ng2 = loss / (np.array(ng) ** 2)
    loss_ng2_visible = loss_ng2[kind[0]:kind[-1]+1]
    loss_ng2_range = np.ptp(loss_ng2_visible)
    ax4.plot(
        wl, loss_ng2,
        color='tab:purple', label=r'Loss/$n_g^2$', linewidth=1.2, linestyle=':'
    )
    ax4.set_ylabel(r'Loss/$n_g^2$ [dB/cm]', color='tab:purple', fontsize=11.5 * text_scale)
    ax4.tick_params(axis='y', which='major', labelsize=10 * text_scale, length=3)
    ax4.set_ylim(
        np.min(loss_ng2_visible) - 0.25 * loss_ng2_range,
        np.max(loss_ng2_visible) + 0.25 * loss_ng2_range
    )

    # ----- Compose legend with custom order -----
    lines = (
        [ax.lines[0]] +
        [ax2.lines[0]] +
        [ax3.lines[0]] +
        [ax4.lines[0]]
    )
    labels = ["Dispersion", "ng", "Loss", "Loss/$n_g^2$"]
    ax.legend(
        lines, labels,
        loc='upper left',
        fontsize=10 * text_scale,
        frameon=False,
        borderaxespad=0.6
    )

    # ------------------ Table Values (Summary Metrics) ------------------
    ng_mean = np.mean(ng[kind[0]:kind[-1]+1])
    Nbw = (max_wl - min_wl) / (min_wl + wl_range/2)
    NDBP = ng_mean * Nbw
    disp_max = np.max(np.abs(disp[kind[0]:kind[-1]]))
    loss_max = np.max(loss_visible)
    max_loss_ng2 = np.max(loss_ng2_visible)
    # Each row is a single value (as a single column)
    cell_text = [
        [f"{ng_mean:.2f}"],
        [f"{Nbw:.3f}"],
        [f"{NDBP:.3f}"],
        [f"{disp_max:.2e}"],
        [f"{loss_max:.2e}"],
        [f"{max_loss_ng2:.2e}"]
    ]
    row_labels = [
        "mean ng",
        "Nbw",
        "NDBP",
        "max |disp|",
        "max loss",
        "max loss/$n_g^2$"
    ]

    # ------------------ Field Imshow ------------------
    field_ax = fig.add_subplot(gs[2])
    ylim = 10 * np.sqrt(3) / 2
    ys = np.linspace(-ylim / 2, ylim / 2, 300)
    mode = 14
    kindex = kind[len(kind)//2]
    z = gme.phc.layers[0].d / 2

    # Get field of original crystal
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] +
                  np.conj(fields['y'])*fields['y'] +
                  np.conj(fields['z'])*fields['z'])

    # Imshow of field
    cax1 = field_ax.imshow(
        eabs,
        extent=[-.5, .5, -ylim/2, ylim/2],
        cmap='plasma'
    )
    # Add circles for inclusions
    for s in gme.phc.layers[0].shapes:
        circle = Circle((s.x_cent, s.y_cent), s.r, edgecolor='white', facecolor='none', linewidth=2)
        surround = Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=2)
        field_ax.add_patch(circle)
        surround.center = (circle.center[0] - np.sign(circle.center[0]), circle.center[1])
        field_ax.add_patch(surround)
    field_ax.set_ylim(-ylim/2, ylim/2)
    field_ax.set_xlim(-0.5, 0.5)
    field_ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False
    )

    # Colorbar below field plot
    cbar_ax = inset_axes(
        field_ax,
        width="70%", height="10%",
        loc='lower center',
        bbox_to_anchor=(-2, -0.1, 5, .75),
        bbox_transform=field_ax.transAxes, borderpad=0
    )
    cb = plt.colorbar(cax1, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'$|\vec{E}|$ (a.u.)', fontsize=10.5 * text_scale)
    cb.ax.tick_params(labelsize=9 * text_scale)

    # ------------------ Rightmost Table Panel ------------------
    table_ax = fig.add_subplot(gs[3])
    table_ax.axis('off')
    table = table_ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center',
        rowLoc='right'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5 * text_scale)
    table.scale(0.95, 1.4)
    table.auto_set_column_width([0, 1])

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight',dpi=300)
    plt.close(fig)

    # calculate the true cost. The l1 difference between the mean ng and the calculated ng
    true_cost = np.sum(np.abs(ng[kind[0]:kind[-1]+1] - ng_mean))/(kind[-1]-kind[0]+1)

    save_dict = {
        'ng':list(ng),
        'loss':list(loss),
        'disp':list(disp),
        'wl':list(wl),
        'kind':list(kind),
        'mean_ng':float(ng_mean),
        'Nbw':float(Nbw),
        'NDBP':float(NDBP),
        'max_disp':float(disp_max),
        'max_loss':float(loss_max),
        'max_loss_ng2':float(max_loss_ng2),
        'vars':vars.tolist(),
        'cost':final_cost,
        'true_cost':float(true_cost),
        'time':float(execution_time),
        'itterations':int(niter)
    }

    with open(path.replace('.png','.json'), 'w') as f:
        json.dump(save_dict, f, indent=4)


# =========================================================
# Utility: encode PNG file to Base64
# =========================================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def runBatchReport(target_ng,target_loss,num_kpoints,path_to_batch,output_path='report'):
    # =========================================================
    # (1) --- Figures and Parameters ---
    # Generate images and parameters from the batch
    # =========================================================

    all_meta = {}   # final output dictionary
    for test_name in os.listdir(path_to_batch):
        test_path = os.path.join(path_to_batch, test_name)

        # Only look at directories (test0, test1, ...)
        if not os.path.isdir(test_path):
            continue

        meta_path = os.path.join(test_path, "meta_data.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

            all_meta[test_name] = meta
    #sort the dictionary to keep an order
    all_meta = dict(sorted(
        all_meta.items(),
        key=lambda item: int(item[0][4:])
        ))

    #pull out the parameters from the data
    ngs,mean_ngs,nbws,max_losses,max_loss_ng2s,max_disps,costs,execution_times,niters,kinds = [],[],[],[],[],[],[],[],[],[]
    for i,data in enumerate(all_meta.values()):
        ngs.append(data['ng'])
        mean_ngs.append(data['mean_ng'])
        nbws.append(data['Nbw'])
        max_losses.append(data['max_loss'])    
        max_loss_ng2s.append(data['max_loss_ng2'])
        max_disps.append(data['max_disp'])
        costs.append(data['cost']) 
        execution_times.append(data['time']) 
        niters.append(data['itterations']) 
        kinds.append(data['kind'])

    #find the true cost
    ngs = np.array(ngs)
    kinds = np.array(kinds)
    mean_ngs = np.array(mean_ngs)
    true_costs = np.sum(np.abs((ngs[:,kinds[0,0]:kinds[0,-1]+1]-mean_ngs[:,None])),axis=1)/(kinds[0,-1]-kinds[0,0]+1)

    #set up the second figure
    fig,ax = plt.subplots(2,2,figsize=(7,5))
    cost_colors = true_costs
    norm = plt.Normalize(cost_colors.min(), cost_colors.max())
    cmap = cm.viridis

    # Find the index of the minimum cost_colors point
    min_idx = np.argmin(cost_colors)

    # Use scatter to color by cost
    sc0 = ax[0,0].scatter(range(len(cost_colors)), cost_colors, c=cost_colors, cmap=cmap, norm=norm)
    ax[0,0].set_xlabel('Test Number')
    ax[0,0].set_ylabel(r'Average $n_g$ Off')
    # Make the minimum point red
    ax[0,0].scatter(min_idx, cost_colors[min_idx], color='red', s=75, zorder=10)

    sc1 = ax[0,1].scatter(niters, np.array(execution_times)/3600, c=cost_colors, cmap=cmap, norm=norm)
    ax[0,1].set_xlabel('Number of Iterations')
    ax[0,1].set_ylabel('Execution Time (hours)')
    # Make the minimum point red
    ax[0,1].scatter(niters[min_idx], execution_times[min_idx]/3600, color='red', s=75, zorder=10)

    sc1 = ax[1,0].scatter(mean_ngs, nbws, c=cost_colors, cmap=cmap, norm=norm)
    ax[1,0].set_xlabel(r'mean $n_g$')
    ax[1,0].set_ylabel('Nbw')
    ax[1,0].scatter(mean_ngs[min_idx], nbws[min_idx], color='red', s=100, zorder=10)

    # Add the lines of constant mean_ngs*nbws=0.2, 0.3, 0.4, 0.5 without affecting axis limits, and annotate each line with its value
    x_min = min(mean_ngs) * 1.05
    x_max = max(mean_ngs) * 1.02
    x_vals = np.linspace(x_min, x_max, 300)
    for const in [0.2, 0.3, 0.4, 0.5]:
        y_vals = const / x_vals
        # Plot the line
        ax[1,0].plot(x_vals, y_vals, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=2, clip_on=False)
        # Add a text label to the left of the line (at the largest x value), slightly offset to the left
        idx = 0  # leftmost x value
        ax[1,0].text(
            x_vals[idx] - 0.03 * (max(mean_ngs)-min(mean_ngs)),   # small left offset
            y_vals[idx],
            f"{const:.1f}",
            color='red',
            fontsize=8,
            verticalalignment='center',
            horizontalalignment='right'
        )

    sc2 = ax[1,1].scatter(max_loss_ng2s, max_disps, c=cost_colors, cmap=cmap, norm=norm)
    ax[1,1].set_xlabel('Max Loss / $n_g^2$')
    ax[1,1].set_ylabel('Max Dispersion')
    ax[1,1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax[1,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[1,1].scatter(max_loss_ng2s[min_idx], max_disps[min_idx], color='red', s=100, zorder=10)

    # Add colorbar to the side for the whole figure
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="7%", pad=0.15)
    cbar = fig.colorbar(sc2, cax=cax)
    cbar.set_label(r"Average $n_g$ Off", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig('tmp.png')
    plt.close(fig)

    #now find min loss and min dispersion index. The true cost must be less than 2.5 for consideration
    masked_loss = np.where(true_costs > target_ng*0.1, np.inf, max_losses)
    masked_disp = np.where(true_costs > target_ng*0.1, np.inf, max_disps)
    min_loss_idx = np.argmin(masked_loss)
    min_disp_idx = np.argmin(masked_disp)

    #encode the images
    img_plot2_64    = encode_image(os.path.join(path_to_batch, f'test{min_idx}', 'meta_data.png'))
    img_summary_64  = encode_image('tmp.png')
    os.remove('tmp.png')
    img_plot2_loss_64 = encode_image(os.path.join(path_to_batch, f'test{min_loss_idx}', 'meta_data.png'))
    img_plot2_disp_64 = encode_image(os.path.join(path_to_batch, f'test{min_disp_idx}', 'meta_data.png'))
    

    #set up last parameters
    min_cost = np.min(costs)
        
    # =========================================================
    # (2) --- HTML Template ---
    # Uses Python .format() substitutions
    # =========================================================

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Test Report</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f9f9f9;
        }}

        h1 {{
            margin-bottom: 10px;
        }}

        table {{
            border-collapse: collapse;
            margin-bottom: 40px;
            width: 100%;
            background: white;
        }}

        th, td {{
            border: 1px solid #ccc;
            padding: 10px 14px;
            text-align: left;
        }}

        th {{
            background: #eee;
            font-weight: bold;
        }}

        .section {{
            margin-bottom: 50px;
            padding: 20px;
            background: white;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}

        .plot {{
            margin-top: 20px;
            margin-bottom: 20px;
        }}
    </style>
    </head>

    <body>

    <h1>Batch Report</h1>

    <table>
        <tr>
            <th>Target ng</th>
            <th>Target loss</th>
            <th>Num Kpoints</th>
            <th>Min Cost</th>
            <th>Min Ng Off</th>
            <th>Total Time (hours)</th>
        </tr>

        <tr>
            <td>{target_ng}</td>
            <td>{target_loss}</td>
            <td>{num_kpoints}</td>
            <td>{min_cost}</td>
            <td>{min_ng_off}</td>
            <td>{total_time}</td>
        </tr>
    </table>

    <div class="section">
        <h2>All Optimization Results</h2>
        <div class="plot">
            <img width="600" src="data:image/png;base64,{img1}">
        </div>
    </div>

    <div class="section">
        <h2>Best Result Summary</h2>
        <div class="plot">
            <p>Smallest Average ng Off. Test: {min_idx}</p>
            <img width="800" src="data:image/png;base64,{img2}">
        </div>
        <div class="plot">
            <p>Smallest Loss. Test: {min_loss_idx}</p>
            <img width="800" src="data:image/png;base64,{img3}">
        </div>
        <div class="plot">
            <p>Smallest Dispersion. Test: {min_disp_idx}</p>
            <img width="800" src="data:image/png;base64,{img4}">
        </div>
    </div>

    </body>
    </html>
    """


    # =========================================================
    # (3) --- Fill Template ---
    # =========================================================

    html_filled = html_template.format(
        target_ng       = round(target_ng,1),
        target_loss     = round(target_loss,3),
        num_kpoints     = num_kpoints,
        min_cost        = round(min_cost,3),
        min_ng_off      = round(np.min(true_costs),3),
        total_time      = round(np.sum(execution_times)/3600,3),
        min_idx         = min_idx,
        min_loss_idx    = min_loss_idx,
        min_disp_idx    = min_disp_idx,
        img1            = img_summary_64,
        img2            = img_plot2_64,
        img3            = img_plot2_loss_64,
        img4            = img_plot2_disp_64,
    )


    # =========================================================
    # (4) --- Write to File ---
    # =========================================================

    with open(output_path+".html", "w") as f:
        f.write(html_filled)

    # =========================================================
    # (4) --- Write to json file ---
    # =========================================================

    output_dict = {
        'target_ng':float(target_ng),
        'target_loss':float(target_loss),
        'num_kpoints':int(num_kpoints),
        'min_cost':float(min_cost),
        'min_ng_off':float(np.min(true_costs)),
        'total_time':float(np.sum(execution_times)/3600),
        'min_idx':int(min_idx),
        'min_loss_idx':int(min_loss_idx),
        'min_disp_idx':int(min_disp_idx),
        'mean_ngs':mean_ngs.tolist(),
        'nbws':nbws,
        'max_losses':max_losses,
        'max_loss_ng2s':max_loss_ng2s,
        'max_disps':max_disps,
        'true_costs':true_costs.tolist(),
    }

    with open(output_path+".json", "w") as f:
        json.dump(output_dict, f, indent=4, sort_keys=True)


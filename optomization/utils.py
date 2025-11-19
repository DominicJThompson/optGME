import numpy as np
import legume
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
    #import backscatter from optomization to avoid circular import
    from optomization import Backscatter
    text_scale = 0.85

    #find the k-points that we optimized for 
    ks = np.linspace(np.pi*.5,np.pi,100)
    kind = [int(np.argmin(np.abs(k-ks))) for k in kpoints]

    #run GME 
    gmeParams = {'verbose':False,'numeig':mode+2,'compute_im':False,'kpoints':np.vstack((ks,[0]*len(ks)))}
    phc = crystal(vars,**phcParams)
    gme = legume.GuidedModeExp(phc,gmax)
    gme.run(**gmeParams)
    cost = Backscatter(a=a,zdiv=10)

    #run ng and loss
    ng = []
    loss = []
    for i in range(len(ks)):
        ng.append(np.abs(NG(gme,i,mode,Nx=100,Ny=125)))
        loss.append(10**cost.cost(gme,phc,mode,k=i)/a/1E-7*10*np.log10(np.e)*ng[-1]**2)
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

def runBatchReport(target_ng,target_loss,num_kpoints,path_to_batch,output_path='report.html'):
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

    #pull out the parameters from the data
    mean_ngs,nbws,max_loss_ng2s,max_disps,costs = [],[],[],[],[]
    for i,data in enumerate(all_meta.values()):
        mean_ngs.append(data['mean_ng'])
        nbws.append(data['Nbw'])
        max_loss_ng2s.append(data['max_loss_ng2'])
        max_disps.append(data['max_disp'])
        costs.append(data['cost'])  

    #set up the second figure
    fig,ax = plt.subplots(2,2,figsize=(7,5))
    cost_colors = np.sqrt(np.array(costs)/num_kpoints)
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

    sc1 = ax[1,0].scatter(mean_ngs, nbws, c=cost_colors, cmap=cmap, norm=norm)
    ax[1,0].set_xlabel(r'mean $n_g$')
    ax[1,0].set_ylabel('Nbw')
    ax[1,0].scatter(mean_ngs[min_idx], nbws[min_idx], color='red', s=100, zorder=10)

    sc2 = ax[1,1].scatter(max_loss_ng2s, max_disps, c=cost_colors, cmap=cmap, norm=norm)
    ax[1,1].set_xlabel('Max Loss / $n_g^2$')
    ax[1,1].set_ylabel('Max Loss')
    ax[1,1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax[1,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[1,1].scatter(max_loss_ng2s[min_idx], max_disps[min_idx], color='red', s=100, zorder=10)

    ax[0,1].axis('off')

    plt.tight_layout()
    plt.savefig('tmp.png')
    plt.close(fig)

    #encode the image
    img_plot2_64    = encode_image(os.path.join(path_to_batch, f'test{min_idx}', 'meta_data.png'))
    img_summary_64  = encode_image('tmp.png')
    os.remove('tmp.png')

    #set up last parameters
    min_cost = costs[min_idx]
        
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
        </tr>

        <tr>
            <td>{target_ng}</td>
            <td>{target_loss}</td>
            <td>{num_kpoints}</td>
            <td>{min_cost}</td>
        </tr>
    </table>

    <div class="section">
        <h2>Best Result Summary</h2>
        <div class="plot">
            <img width="600" src="data:image/png;base64,{img2}">
        </div>
    </div>

    <div class="section">
        <h2>All Optimization Results</h2>
        <div class="plot">
            <img width="600" src="data:image/png;base64,{img1}">
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
        img1            = img_summary_64,
        img2            = img_plot2_64,
    )


    # =========================================================
    # (4) --- Write to File ---
    # =========================================================

    with open(output_path, "w") as f:
        f.write(html_filled)

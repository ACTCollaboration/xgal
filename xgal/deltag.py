"""
xgal.deltag

Contains functions and classes for converting catalogs
to overdensity maps.

"""
import numpy as np
import healpy as hp

def get_catalog(version_tag=None,sim_index=None):
    pass

def make_healpix_map(ra, dec, quantity, nside, mask=None, weight=None, fill_UNSEEN=False, return_extra=False):
    """
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the average of quantity in each pixel.

    Parameters
    ----------
    ra : array
        Right ascension.
    dec : array
        Declination.
    quantity : array
        `quantity` can be 2D, in which case several maps are created or None, in
        which case only count maps are returned.
    nside : int
        `nside` parameter for healpix.
    mask : array
        If None, the mask is created and has value 1 in pixels that contain at
        least one object, 0 elsewhere.
    weight : type
        Weights of objects (the default is None, in which case all objects have
        weight 1). Must be the same size as `quantity`.
    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with
        hp.UNSEEN, 0 otherwise (the default is False).
    return_extra : boolean
        If True, a dictionnary is returned that contains count statistics and
        the masked `ipix` array to allow for statistics on the quantities to be
        computed.

    Returns
    -------
    List of outmaps, the count map and the mask map.

    """

    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
            assert np.all(weight > 0.), "[make_healpix_map] weight is not strictly positive"
            weight = np.atleast_2d(weight)
        else:
            weight = np.ones_like(quantity)

        assert quantity.shape[1] == weight.shape[1], "[make_healpix_map] quantity/weight arrays don't have the same length"

        assert len(ra) == len(dec), "[make_healpix_map] ra/dec arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask)==npix, "[make_healpix_map] mask array does not have the right length"

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    count = np.zeros(npix, dtype=float)
    outmaps = []

    # Getting pixels for each object
    ipix = hp.ang2pix(nside, (90.0-dec)/180.0*np.pi, ra/180.0*np.pi)

    # Counting objects in pixels
    np.add.at(count, ipix, 1.)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # Masking the count in the masked area
    count[np.logical_not(bool_mask)] = x
    if mask is None:
        assert np.all(count[bool_mask] > 0), "[make_healpix_map] count[bool_mask] is not positive on the provided mask !"

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            np.add.at(sum_w, ipix, weight[i,:])

            outmap = np.zeros(npix, dtype=float)
            np.add.at(outmap, ipix, quantity[i,:]*weight[i,:])
            outmap[bool_mask] /= sum_w[bool_mask]
            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)

    if mask is None:
        returned_mask = bool_mask.astype(float)
    else:
        returned_mask = mask

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1. / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        return outmaps, count, returned_mask, extra
    else:
        return outmaps, count, returned_mask
#

def count2density(count, mask_frac=None, mask=None):
    """
    Creates a reconstructed density map from count-in-pixel map count, with
    completeness and mask support.

    Parameters
    ----------
    count : array
        Healpix map of number count of object per pixel.
    mask_frac : array (optional)
        Healpix map of the fraction each pixel has been observed, also called
        completeness or masked map fraction (the default is None).
    mask : array
        Binary mask of the sky (the default is None).

    Returns
    -------
    array
        Density map.

    """

    npix = len(count)

    if mask_frac is None:
        mask_frac = np.ones(npix, dtype=float)
    if mask is None:
        mask = np.ones(npix, dtype=bool)

    msk = mask.astype(bool)

    # Local mean density to compare count with.
    avg_in_pixel = np.zeros(npix, dtype=float)
    avg_in_pixel[msk] = mask_frac[msk] * np.sum(count[msk]) / np.sum(mask_frac[msk])

    # Density
    density = np.zeros(npix, dtype=float)
    density[msk] = count[msk] / avg_in_pixel[msk] - 1.

    return density
#

def density2count(densitymap, nbar, mask=None, completeness=None, pixel=True):
    """
    Generates a Poisson sampling of a given density map.

    Parameters
    ----------
    densitymap : array
        Healpix map of density field. Note that the density map is clipped where it is below -1.
    nbar : float
        Mean galaxy density.
    mask : array (optional)
        Binary mask of where to perform sampling.
    pixel : type
        If pixel=True (default), nbar is the mean number of galaxies per pixel.
        If pixel=False, nbar is the angular density (per unit steradian).
    Returns
    -------
    array
        Map with the number of object per pixel.

    """
    if mask is None:
        mask = np.ones(len(densitymap), dtype=bool)
    if completeness is None:
        completeness = mask.astype(float)

    if np.any(densitymap[mask] < - 1.):
        print("[density2count] The density map has pixels below -1, will be clipped.")

    if pixel :
        nbarpix = nbar
    else :
        nbarpix = nbar * hp.nside2pixarea(hp.npix2nside(len(densitymap)))

    onepdelta = np.clip(1. + densitymap, 0., np.inf)
    onepdelta[np.logical_not(mask.astype(bool))] = 0.

    lamb = nbarpix * onepdelta * completeness

    return np.random.poisson(lamb)
#

def overdensity(ras,decs,mask_frac,nside):
    binary_mask = (mask_frac==0.0)

    _, count, _ = make_healpix_map(ras, decs, quantity, nside, mask=binary_mask, weight=None, fill_UNSEEN=False, return_extra=False)

    density = count2density(count, mask_frac=mask_frac, mask=binary_mask)

    return density

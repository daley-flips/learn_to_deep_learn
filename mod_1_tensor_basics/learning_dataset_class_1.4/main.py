from dataset import toy_set
from transforms.add_mult import add_mult
from transforms.mult import mult
print('\n')
# =============================================================================
# =============================================================================
def dataset_fucntions():
    # create object
    print('calling the init function')
    ds = toy_set()
    print(ds)
    
    # first element
    print('\ncalling the getitem function (works like arrays)')
    print(ds[0])
    
    print('\ncalling len')
    print(ds.len)
    
    print('\nthis is how to loop through the dataset')
    for i in range(3):
        x, y = ds[i]
        print('x:', x)
        print('y', y)
        print()
# =============================================================================
# =============================================================================
def transform_intro():
    ds = toy_set()
    am = add_mult()
    
    x, y = ds[0]
    print('x:', x)
    print('y', y)
    
    x_, y_ = am(ds[0])
    
    print('x_:', x_)
    print('y_', y_)
# =============================================================================
# ============================================================================= 
def better_way_to_transform():
    ds = toy_set()
    am = add_mult()

    print(ds[0])
    print('\napply transform')
    ds_ = toy_set(transform=am)
    print(ds_[0])
    print('a transform can be performed simply by indexing because thats how we implemented __getitem__')
# =============================================================================
# =============================================================================
from torchvision import transforms
# ^ gotta add that import
def multiple_transforms():
    ds = toy_set()
    x, y = ds[0]
    print('x:', x)
    print('y', y)
    
    print('\ndouble transform time:')
    
    double_t = transforms.Compose([add_mult(), mult()])
    x_, y_ = double_t(ds[0])
    print('x_:', x_)
    print('y_', y_)
    
    
# =============================================================================
# =============================================================================


# dataset_fucntions()
# transform_intro()
# better_way_to_transform()
multiple_transforms()

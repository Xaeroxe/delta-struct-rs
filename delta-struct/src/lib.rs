pub use delta_struct_macros::Delta;

pub trait Delta {
    type Output;

    fn delta(old: Self, new: Self) -> Option<Self::Output>;
}
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Delta)]
    struct UnitType;

    #[derive(Delta)]
    // #[delta_struct(attributes(derive(Clone, Debug)))]
    struct NewType(i32);

    #[derive(Delta)]
    struct NewTypeWithGeneric<T>(T);

    #[derive(Delta)]
    struct SimpleType {
        foo: i32,
        bar: bool,
    }

    #[derive(Delta)]
    struct SimpleTypeWithGeneric<T> {
        foo: T,
        bar: bool,
    }

    #[derive(Delta)]
    struct SimpleCollectionWithGeneric<T> {
        #[delta_struct(field_type = "unordered")]
        foo: Vec<T>,
        bar: bool,
    }

    #[derive(Delta)]
    struct DeltaRecursion {
        #[delta_struct(field_type = "delta")]
        foo: NewType,
        bar: bool,
    }


    #[derive(Delta)]
    #[delta_struct(default = "unordered")]
    struct AttributeTest {
        #[delta_struct(field_type = "scalar")]
        foo: i32,
        #[delta_struct(field_type = "scalar")]
        bar: i32,
        baz: Vec<i32>,
    }

    #[test]
    fn it_works() {
        let old = SimpleCollectionWithGeneric {
            foo: vec![1, 2, 3],
            bar: false,
        };
        let new = SimpleCollectionWithGeneric {
            foo: vec![3, 4, 5],
            bar: true,
        };
        let delta = Delta::delta(old, new).unwrap();
        assert_eq!(delta.foo_add, vec![4, 5]);
        assert_eq!(delta.foo_remove, vec![1, 2]);
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn delta_false_positive_check() {
        let old = NewType(5);
        let new = NewType(5);
        let delta = Delta::delta(old, new);
        assert!(delta.is_none());
    }

    #[test]
    fn scalar_delta_false_positive_check() {
        let old = SimpleType { foo: 5, bar: false };
        let new = SimpleType { foo: 5, bar: true };
        let delta = Delta::delta(old, new).unwrap();
        assert!(delta.foo.is_none());
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn delta_field_test() {
        let old = DeltaRecursion { foo: NewType(5), bar: false };
        let new = DeltaRecursion { foo: NewType(6), bar: true };
        let delta = Delta::delta(old, new).unwrap();
        // TODO: Use assert_eq when we build out delta struct
        // attributes.
        if let Some(NewTypeDelta { field_0: Some(6) }) = delta.foo {
           // Do nothing, this is the pass case. 
        } else {
            panic!();
        }
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn default_type_respected() {
        let old = AttributeTest {
            foo: 5,
            bar: 4,
            baz: vec![],
        };
        let new = AttributeTest {
            foo: 5,
            bar: 4,
            baz: vec![9, 4, 5],
        };
        let delta = Delta::delta(old, new).unwrap();
        assert!(delta.foo.is_none());
        assert!(delta.bar.is_none());
        assert_eq!(delta.baz_add, vec![9, 4, 5]);
        assert_eq!(delta.baz_remove, vec![]);
    }
}

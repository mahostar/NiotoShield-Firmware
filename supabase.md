create table public.products (
  id uuid not null default gen_random_uuid (),
  product_key text not null,
  used boolean null default false,
  user_id uuid null,
  public_key text null,
  admin_id uuid null,
  constraint products_pkey primary key (id),
  constraint products_product_key_key unique (product_key),
  constraint products_user_id_key unique (user_id),
  constraint products_admin_id_fkey foreign KEY (admin_id) references admins (id) on update CASCADE on delete set null,
  constraint products_user_id_fkey foreign KEY (user_id) references users (id) on update CASCADE on delete set null,
  constraint products_user_id_fkey1 foreign KEY (user_id) references users (id) on update CASCADE on delete set null
) TABLESPACE pg_default;


create table public.localisations (
  id uuid not null default extensions.uuid_generate_v4 (),
  user_id uuid not null,
  product_key text not null,
  lat double precision null,
  lng double precision null,
  created_at timestamp with time zone null default now(),
  constraint localisations_pkey primary key (id),
  constraint fk_product_key foreign KEY (product_key) references products (product_key) on delete CASCADE,
  constraint fk_user_id foreign KEY (user_id) references users (id) on delete CASCADE
) TABLESPACE pg_default;